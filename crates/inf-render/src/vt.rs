//! **The virtual-texture GPU mirror** (P26.2): one physical page atlas, one
//! indirection buffer, and the code that applies an [`inf_vt::VtTransaction`] to
//! them.
//!
//! This module owns `wgpu` and nothing else. Every decision about *which* page
//! lives *where* was made in `inf-vt`, with no adapter, and is unit-tested there;
//! what is left here is plumbing — and plumbing is exactly what the arms in
//! `tests/vt_pools.rs` read back byte for byte.
//!
//! **Nothing renders through this yet.** No pass binds it, no shader names it,
//! and `inf_render::renderer` does not know it exists. P26.3 binds it into the
//! lit passes; this batch exists so that when it does, the atlas, the upload
//! pitch and the table bytes underneath it are already known-good.
//!
//! # The ordering contract P26.3 relies on
//!
//! A transaction is applied **whole, before any frame samples it**. That is not a
//! hope about call order — it is a property of `wgpu`'s queue: `write_texture`
//! and `write_buffer` are staged on the queue and execute **before** the commands
//! of any command buffer submitted afterwards. So:
//!
//! * call [`VtPools::apply`] at the frame's sync point, at any time before
//!   `queue.submit` of that frame's encoder;
//! * never call it *between* two submits that both sample the atlas.
//!
//! Follow those two and a frame sees the state before a transaction or the state
//! after it, never a half. This is the same argument
//! `inf_vgeom::stream`'s "pools grew ⇒ re-stage everything" comment makes for
//! buffers, and it is why an evict costs nothing on the GPU: no live indirection
//! entry names an evicted slot, so its texels are unreachable until the admit
//! that overwrites them — and that admit is in this same transaction or a later
//! one, never in between.
//!
//! # The row pitch, measured rather than assumed
//!
//! A 136-texel BC1 tile is 34 blocks × 8 B = **272 B per block-row**, and 272 is
//! not a multiple of 256. That matters for `copy_buffer_to_texture`, which
//! requires `bytes_per_row` to be a multiple of
//! [`wgpu::COPY_BYTES_PER_ROW_ALIGNMENT`] — and **does not matter here**, because
//! `Queue::write_texture` takes the *tight* pitch and re-packs into its own
//! staging belt. A cooked tile therefore uploads exactly as it lies in the mmap,
//! with no repacking on our side and no padded staging buffer of our own. (The
//! P26.1 upload proof measured this for a standalone texture; `tests/vt_pools.rs`
//! measures it again into a **sub-region** of an atlas, which is the case that
//! actually ships.)
//!
//! The tight pitch is not merely *permitted* here, it is **required**: padding
//! `bytes_per_row` to 256 without padding the data makes `wgpu` reject the write
//! outright, because the declared layout then describes more bytes than the slice
//! holds. Measured by mutation, not assumed.
//!
//! A slot origin is `slot_index × 136` texels, and 136 is a multiple of 4, so
//! every origin is BC-block aligned — the one alignment rule `write_texture`
//! really does enforce on a block format.

use std::borrow::Cow;

use inf_vt::{PageFormat, VtAdmit, VtResidency, VtTransaction};

use crate::settings::RenderSettings;

/// Bindings the VT surface needs. P26.3 folds exactly this many into the shared
/// environment group, which is why the indirection is one storage buffer holding
/// every texture's table rather than a texture per texture — see
/// [`inf_vt::table`].
pub const VT_BINDING_COUNT: u32 = 3;

/// The `wgpu` format a page uploads as.
///
/// **The P26.1 remainder, discharged.** It lived in `tests/vt_bc_upload.rs`
/// because there was no crate to put it in; the crate is `inf-vt`, the mapping
/// needs `wgpu`, and `inf-vt` is GPU-free — so it lands in the mirror, which is
/// the half of the seam that owns `wgpu`. That test now *calls* this rather than
/// keeping its copy: a mapping in two places is a mapping that drifts, and the
/// copy in a test is the one that drifts unnoticed.
pub fn page_format(format: PageFormat, srgb: bool) -> wgpu::TextureFormat {
    match (format, srgb) {
        (PageFormat::Rgba8, true) => wgpu::TextureFormat::Rgba8UnormSrgb,
        (PageFormat::Rgba8, false) => wgpu::TextureFormat::Rgba8Unorm,
        (PageFormat::Bc1, true) => wgpu::TextureFormat::Bc1RgbaUnormSrgb,
        (PageFormat::Bc1, false) => wgpu::TextureFormat::Bc1RgbaUnorm,
        (PageFormat::Bc3, true) => wgpu::TextureFormat::Bc3RgbaUnormSrgb,
        (PageFormat::Bc3, false) => wgpu::TextureFormat::Bc3RgbaUnorm,
        // **Neither of Wave T's formats has an sRGB variant, and neither wants
        // one.** BC5 stores a normal's X and Y — a transfer curve on a direction
        // is meaningless — and RGBA16F is linear by construction. The `srgb`
        // argument is not ignored for them so much as answered: the atlas has
        // always been created in the LINEAR variant and the shader decodes per
        // texture from the table's flag (see `vt_sample.wgsl`'s header), so
        // there was never a second view to pick.
        (PageFormat::Bc5, _) => wgpu::TextureFormat::Bc5RgUnorm,
        (PageFormat::Rgba16F, _) => wgpu::TextureFormat::Rgba16Float,
    }
}

/// The format a pool takes, given what the content is stored as and what the
/// adapter was granted.
///
/// **The first reader of `RenderSettings::vt.bc_tiles`.** P26.1 added the
/// setting, the probe and the clamp in both hosts and left the ledger honest
/// about it: "`vt.bc_tiles` has no reader". This is it. An adapter without
/// `TEXTURE_COMPRESSION_BC` has the flag clamped off by
/// [`AdapterCaps::clamp_bc_tiles`](crate::caps::AdapterCaps::clamp_bc_tiles), and
/// the pool becomes RGBA8 — the caller then pages the same tiles through
/// `TiledTextureReader::tile_rgba8` instead of `tile`, which is **the same
/// residency door, one format decision earlier**, at 8× the page bytes for BC1
/// and 4× for BC3.
/// **The clamp is on the BC formats, not on "everything but RGBA8"** (Wave T).
/// `PageFormat::Rgba16F` needs no `TEXTURE_COMPRESSION_BC` — it is an
/// uncompressed float format — so an adapter without BC keeps it whole rather
/// than being handed a transcode that would clamp exactly the range the format
/// exists to carry. `PageFormat::needs_bc` is the one place that decision lives;
/// `TiledTextureReader::tile_rgba8` refuses a float page for the same reason, so
/// the two ends of this seam cannot drift into disagreeing.
pub fn pool_format(settings: &RenderSettings, stored: PageFormat) -> PageFormat {
    if settings.vt.bc_tiles || !stored.needs_bc() {
        stored
    } else {
        PageFormat::Rgba8
    }
}

/// What one [`VtPools::apply`] did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VtApplyReport {
    /// Pages written into the atlas.
    pub pages: usize,
    /// Bytes of page data written.
    pub page_bytes: u64,
    /// Bytes of indirection written.
    pub table_bytes: u64,
    /// Whether the indirection buffer was re-created (a texture registered).
    pub table_recreated: bool,
    /// Admits whose bytes the caller could not produce.
    ///
    /// Their slots are filled with a **deterministic zero page** rather than left
    /// holding whatever the previous occupant put there: an indirection entry
    /// already points at the slot, so "leave it alone" would sample another
    /// texture's texels. Zero is black (or transparent black) — visibly nothing,
    /// which is the honest rendering of "this page did not load", and it is
    /// reported here so a caller can drop the texture rather than ship it.
    pub missing: Vec<VtAdmit>,
}

/// The physical page atlas and the indirection buffer for one
/// [`inf_vt::VtResidency`].
pub struct VtPools {
    atlas: wgpu::Texture,
    atlas_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    table: wgpu::Buffer,
    format: wgpu::TextureFormat,
    geometry: inf_vt::VtPoolGeometry,
    page_bytes: u64,
    /// The residency layout generation this buffer was built for. **A stamp, not
    /// a measurement** — the only legal operation on it is `!=`.
    layout_generation: u64,
    /// Bumped every time [`table`](Self::table) becomes a **different buffer**.
    ///
    /// P26.3's `EnvBinding` caches a bind group that names that buffer, so a
    /// re-creation it cannot see is a bind group that keeps the old allocation
    /// alive and keeps sampling a table from before the registration —
    /// silently, because `wgpu` refcounts the old buffer and validates nothing.
    /// This is the component of `passes::ResourceKey` that makes it visible.
    /// **A stamp, not a measurement.**
    table_generation: u64,
    /// One zero page, allocated once, for the missing-fetch path.
    zero_page: Vec<u8>,
}

/// The **empty** VT surface: what the environment bind group names on a frame
/// with no pool at all.
///
/// Created once in `EngineRenderer::new` and never recreated (so it contributes
/// nothing to [`crate::passes::ResourceKey`], on the `wetness` precedent). A
/// layout entry must be satisfied whether or not anything is textured, and the
/// three resources here are chosen so that being satisfied reads as *nothing*:
/// the table buffer's first word is **zero, not [`inf_vt::TABLE_MAGIC`]**, so
/// `vt_active()` in `vt_sample.wgsl` is false and every lit shader takes the
/// scalar path without touching the atlas. That is the structural half of "a
/// textureless scene renders exactly as it did"; the other half is that no
/// instance names a texture, and both have to hold.
pub struct VtEmptyPool {
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    table: wgpu::Buffer,
}

impl VtEmptyPool {
    pub fn new(device: &wgpu::Device) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vt-atlas-absent"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Filterable float, to match the layout entry the real atlas fills.
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        Self {
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("vt-sampler-absent"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            // Four zero bytes: never written, so word 0 is 0 and not the magic.
            table: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vt-indirection-absent"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
        }
    }

    #[inline]
    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }
    #[inline]
    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }
    #[inline]
    pub fn table(&self) -> &wgpu::Buffer {
        &self.table
    }
}

impl VtPools {
    /// Create the atlas and the indirection buffer for `residency`.
    ///
    /// `srgb` picks between the linear and sRGB view of the page format. It is a
    /// **pool-wide** decision, not a per-texture one, because there is one atlas.
    ///
    /// **P26.3 resolved that remainder, and the answer is `false`.** The renderer
    /// always creates the LINEAR pool and `vt_sample.wgsl` applies the sRGB
    /// transfer function per texture, from the `srgb` flag already sitting in
    /// that texture's block of the indirection table. Two pools (3 → 6 bindings,
    /// two budgets, two residencies) and a `view_formats` reinterpretation
    /// (4 bindings, and the shader still has to read the flag to pick a view)
    /// were both weighed and both cost a binding or more to express one bit that
    /// is already in the table. The cost of the answer taken is that the transfer
    /// function lands on the FILTERED result rather than per texel — ordinary
    /// gamma-space filtering error, written down in the shader's header.
    ///
    /// The parameter stays because it is real — the mapping is total over six
    /// format pairs and `tests/vt_pools.rs` exercises both — and because a debug
    /// view that wants the hardware decode has somewhere to ask for it.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        residency: &VtResidency,
        srgb: bool,
    ) -> Self {
        let geometry = residency.geometry();
        let format = page_format(residency.config().format, srgb);
        // A pool whose budget bought no slots still has to produce a legal
        // texture; it can hold nothing, so one empty slot costs nothing and keeps
        // the type total.
        let extent = wgpu::Extent3d {
            width: geometry.atlas_width().max(geometry.stored_tile_size),
            height: geometry.atlas_height().max(geometry.stored_tile_size),
            depth_or_array_layers: 1,
        };
        let atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vt-page-atlas"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            // COPY_SRC so the residency heat-map and the readback arms can look
            // at a page; the shipping path only ever writes and samples.
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let atlas_view = atlas.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("vt-page-sampler"),
            // CLAMP, and the border ring is what makes that correct: a texel at a
            // tile's payload edge filters against real neighbours baked into the
            // page, so clamping at the SLOT edge is never reached by a legal
            // sample. Repeat would wrap into the next slot.
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            // The atlas has no mip chain: a page IS a mip level's tile, and the
            // pyramid lives in the virtual address space, not in the texture.
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let (table, _) = upload_whole_table(device, queue, residency);
        Self {
            atlas,
            atlas_view,
            sampler,
            table,
            format,
            geometry,
            page_bytes: residency.page_bytes(),
            layout_generation: residency.layout_generation(),
            table_generation: residency.layout_generation(),
            zero_page: vec![0u8; residency.page_bytes() as usize],
        }
    }

    /// The generation of the **indirection buffer allocation** — it moves iff
    /// [`table`](Self::table) becomes a different buffer. See the field docs for
    /// what reads it and what goes wrong without it.
    #[inline]
    pub fn table_generation(&self) -> u64 {
        self.table_generation
    }

    /// The atlas texture (for a debug view or a readback).
    #[inline]
    pub fn atlas(&self) -> &wgpu::Texture {
        &self.atlas
    }
    #[inline]
    pub fn atlas_view(&self) -> &wgpu::TextureView {
        &self.atlas_view
    }
    #[inline]
    pub fn sampler(&self) -> &wgpu::Sampler {
        &self.sampler
    }
    /// The indirection buffer — every registered texture's table, at the layout
    /// [`inf_vt::table`] documents.
    #[inline]
    pub fn table(&self) -> &wgpu::Buffer {
        &self.table
    }
    #[inline]
    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }
    #[inline]
    pub fn geometry(&self) -> inf_vt::VtPoolGeometry {
        self.geometry
    }

    /// The three bind-group layout entries P26.3 folds into the environment
    /// group, starting at `base`.
    pub fn bind_group_layout_entries(base: u32) -> [wgpu::BindGroupLayoutEntry; 3] {
        [
            wgpu::BindGroupLayoutEntry {
                binding: base,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: base + 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: base + 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ]
    }

    /// The matching bind-group entries.
    pub fn bind_group_entries(&self, base: u32) -> [wgpu::BindGroupEntry<'_>; 3] {
        [
            wgpu::BindGroupEntry {
                binding: base,
                resource: wgpu::BindingResource::TextureView(&self.atlas_view),
            },
            wgpu::BindGroupEntry {
                binding: base + 1,
                resource: wgpu::BindingResource::Sampler(&self.sampler),
            },
            wgpu::BindGroupEntry {
                binding: base + 2,
                resource: self.table.as_entire_binding(),
            },
        ]
    }

    /// **Apply a transaction, whole.**
    ///
    /// `fetch` produces one admitted page's bytes in the pool's format — a
    /// borrowed mmap slice (`TiledTextureReader::tile`) on a BC pool, an owned
    /// transcode (`tile_rgba8`) on the fallback. `Cow` is the type that says
    /// exactly that, and it is why both arms are one door.
    ///
    /// Evicts do nothing here on purpose (see the module docs). Table blocks are
    /// written for exactly the textures the transaction names, unless the layout
    /// moved — then the buffer is re-created and written whole, because a new
    /// texture shifts every block after it.
    pub fn apply<'a>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        residency: &VtResidency,
        txn: &VtTransaction,
        mut fetch: impl FnMut(&VtAdmit) -> Option<Cow<'a, [u8]>>,
    ) -> VtApplyReport {
        let mut report = VtApplyReport::default();

        // ── 1. pages ──
        let (bw, bh) = self.format.block_dimensions();
        let block = self
            .format
            .block_copy_size(None)
            .expect("a colour format has a block size");
        let side = self.geometry.stored_tile_size;
        let blocks_x = side.div_ceil(bw);
        let blocks_y = side.div_ceil(bh);
        let tight = blocks_x * block;
        for admit in &txn.admits {
            let Some((ox, oy)) = self.geometry.slot_origin(admit.slot) else {
                // A slot the residency handed out is inside the geometry the
                // residency was built from; a mismatch means the two are out of
                // step, which is a caller bug rather than a page fault.
                tracing::error!(
                    "inf-render: VT admit names slot {} outside a {}×{} atlas",
                    admit.slot,
                    self.geometry.slots_x,
                    self.geometry.slots_y
                );
                report.missing.push(*admit);
                continue;
            };
            let bytes = match fetch(admit) {
                Some(b) if b.len() as u64 == self.page_bytes => b,
                other => {
                    if let Some(b) = other {
                        tracing::warn!(
                            "inf-render: VT page for slot {} is {} B, not the pool's {} B",
                            admit.slot,
                            b.len(),
                            self.page_bytes
                        );
                    }
                    report.missing.push(*admit);
                    Cow::Borrowed(self.zero_page.as_slice())
                }
            };
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.atlas,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: ox, y: oy, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                &bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    // The TIGHT pitch — see the module docs on why 272 B is legal
                    // here and not in `copy_buffer_to_texture`.
                    bytes_per_row: Some(tight),
                    rows_per_image: Some(blocks_y),
                },
                wgpu::Extent3d {
                    width: side,
                    height: side,
                    depth_or_array_layers: 1,
                },
            );
            report.pages += 1;
            report.page_bytes += self.page_bytes;
        }

        // ── 2. indirection ──
        if txn.layout_rebuilt || residency.layout_generation() != self.layout_generation {
            let (table, bytes) = upload_whole_table(device, queue, residency);
            self.table = table;
            self.layout_generation = residency.layout_generation();
            // A DIFFERENT allocation, so anything caching a bind group over it
            // has to be told. The residency's layout stamp is a global monotone
            // counter, so this never repeats a value a stale cache may hold.
            self.table_generation = residency.layout_generation();
            self.zero_page = vec![0u8; residency.page_bytes() as usize];
            self.page_bytes = residency.page_bytes();
            report.table_bytes = bytes;
            report.table_recreated = true;
        } else {
            for tex in &txn.tables {
                let Some((base, words)) = residency.table_block(*tex) else {
                    continue;
                };
                queue.write_buffer(&self.table, (base * 4) as u64, bytemuck::cast_slice(words));
                report.table_bytes += (words.len() * 4) as u64;
            }
        }
        report
    }
}

/// Buffer bytes for `words` table words, never zero (a `wgpu` buffer may not be)
/// and always a multiple of 4.
fn table_buffer_len(words: usize) -> u64 {
    (words.max(1) * 4) as u64
}

/// A fresh indirection buffer holding the whole table. Returns it and the bytes
/// written.
///
/// Written with `queue.write_buffer`, the same door the per-frame block writes
/// use — so there is one upload path to reason about rather than two, and the
/// ordering argument in the module docs covers both. It runs at pool creation and
/// on a registration, never inside the frame loop, so the staging copy is not on
/// any hot path.
fn upload_whole_table(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    residency: &VtResidency,
) -> (wgpu::Buffer, u64) {
    let words = residency.table_words();
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vt-indirection"),
        size: table_buffer_len(words.len()),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytemuck::cast_slice(words));
    (buffer, (words.len() * 4) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every page format maps to a `wgpu` format whose block copy size is the
    /// page's own arithmetic — the mapping and `inf_vt::PageFormat::page_bytes`
    /// cannot disagree without this failing.
    #[test]
    fn the_page_format_mapping_agrees_with_the_page_size() {
        for (fmt, srgb) in [
            (PageFormat::Rgba8, false),
            (PageFormat::Rgba8, true),
            (PageFormat::Bc1, false),
            (PageFormat::Bc1, true),
            (PageFormat::Bc3, false),
            (PageFormat::Bc3, true),
        ] {
            let wgpu_fmt = page_format(fmt, srgb);
            let (bw, bh) = wgpu_fmt.block_dimensions();
            let block = wgpu_fmt.block_copy_size(None).expect("colour format");
            let side = 136u32;
            let expect = u64::from(side.div_ceil(bw) * side.div_ceil(bh) * block);
            assert_eq!(
                fmt.page_bytes(side),
                expect,
                "{fmt:?} srgb={srgb} → {wgpu_fmt:?}"
            );
            assert_eq!(wgpu_fmt.is_srgb(), srgb, "{fmt:?}");
        }
        // Wave T's two formats, on the same arithmetic. They are excluded from
        // the sRGB half of the sweep above deliberately and the exclusion is
        // asserted here: neither has an sRGB variant, and neither wants one — a
        // transfer curve on a normal's X and Y is meaningless, and RGBA16F is
        // linear by construction.
        for fmt in [PageFormat::Bc5, PageFormat::Rgba16F] {
            for srgb in [false, true] {
                let wgpu_fmt = page_format(fmt, srgb);
                let (bw, bh) = wgpu_fmt.block_dimensions();
                let block = wgpu_fmt.block_copy_size(None).expect("colour format");
                let side = 136u32;
                assert_eq!(
                    fmt.page_bytes(side),
                    u64::from(side.div_ceil(bw) * side.div_ceil(bh) * block),
                    "{fmt:?} → {wgpu_fmt:?}"
                );
                assert!(!wgpu_fmt.is_srgb(), "{fmt:?} must have no sRGB variant");
            }
        }
        // The claim the BC5 path exists for, in wgpu's own arithmetic.
        assert_eq!(
            PageFormat::Rgba8.page_bytes(136) / PageFormat::Bc5.page_bytes(136),
            4
        );
    }

    /// **A float pool is NOT clamped by the BC feature** (Wave T), and
    /// **trilinear is off by default** (Wave T's golden constraint).
    #[test]
    fn wave_t_defaults_leave_the_frame_exactly_as_it_was() {
        let mut settings = RenderSettings::default();
        assert!(
            !settings.vt.trilinear,
            "trilinear on by default would move every textured golden"
        );
        // An adapter without BC transcodes the block formats and leaves the
        // float one alone — flattening an RGBA16F page to RGBA8 is precisely the
        // silent loss the format was added to end, and a mobile adapter is the
        // one class that cannot argue back.
        settings.vt.bc_tiles = false;
        assert_eq!(
            pool_format(&settings, PageFormat::Rgba16F),
            PageFormat::Rgba16F
        );
        assert_eq!(pool_format(&settings, PageFormat::Bc5), PageFormat::Rgba8);
        settings.vt.bc_tiles = true;
        assert_eq!(pool_format(&settings, PageFormat::Bc5), PageFormat::Bc5);
    }

    /// The clamp's first reader does what the clamp meant.
    #[test]
    fn a_clamped_adapter_takes_the_rgba8_pool() {
        let mut settings = RenderSettings::default();
        assert!(settings.vt.bc_tiles, "BC is the default");
        assert_eq!(pool_format(&settings, PageFormat::Bc1), PageFormat::Bc1);
        assert_eq!(pool_format(&settings, PageFormat::Bc3), PageFormat::Bc3);
        settings.vt.bc_tiles = false;
        for stored in [PageFormat::Bc1, PageFormat::Bc3, PageFormat::Rgba8] {
            assert_eq!(
                pool_format(&settings, stored),
                PageFormat::Rgba8,
                "{stored:?} must transcode when the adapter has no BC"
            );
        }
    }
}
