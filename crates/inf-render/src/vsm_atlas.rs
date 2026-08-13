//! **The virtual-shadow-map GPU mirror** (P27.1): one physical `Depth32Float`
//! page atlas, one indirection buffer, and the code that applies an
//! [`inf_vsm::VsmTransaction`] to them.
//!
//! This module owns `wgpu` and nothing else. Every decision about *which* page
//! lives *where* was made in `inf-vsm`, with no adapter, and is unit-tested
//! there; what is left here is plumbing — and plumbing is what
//! `tests/vsm_marking.rs` reads back byte for byte.
//!
//! # An admit has no bytes, and that is the difference from `inf_render::vt`
//!
//! `VtPools::apply` fetches each admitted page's bytes and uploads them. A
//! shadow page has no bytes to fetch: its content is **rasterized** by P27.2's
//! per-page caster pass, into the slot this transaction just published. So
//! `apply` here writes the *table* and nothing else, and an admit is an
//! allocation rather than an upload.
//!
//! That is also why the atlas is created `RENDER_ATTACHMENT | TEXTURE_BINDING`
//! rather than `COPY_DST | TEXTURE_BINDING`: nothing ever writes into it from
//! the CPU, and P27.2 will `set_viewport` / `set_scissor` onto a slot's origin
//! (which is what [`inf_vsm::VsmAtlasGeometry::slot_origin`] is for) and draw.
//!
//! # The ordering contract
//!
//! Identical to `crate::vt`'s, and for the same reason — `write_buffer` is
//! staged on the queue and executes before the commands of any command buffer
//! submitted afterwards. Call [`VsmPools::apply`] at the frame's sync point, at
//! any time before that frame's `queue.submit`; never between two submits that
//! both read the table.

use inf_vsm::{VsmResidency, VsmTransaction};

/// The `wgpu` format a shadow page is stored in. `Depth32Float`, always — see
/// [`inf_vsm::VSM_PAGE_BYTES_PER_TEXEL`] for why there is no format decision.
pub const VSM_PAGE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// What one [`VsmPools::apply`] did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VsmApplyReport {
    /// Pages that gained a slot — **allocations, not uploads** (see the module
    /// docs). P27.2 is what turns each one into a raster.
    pub pages: usize,
    /// Pages whose slots became unreachable.
    pub evicted: usize,
    /// Bytes of indirection written.
    pub table_bytes: u64,
    /// Whether the indirection buffer was re-created (a light was registered).
    pub table_recreated: bool,
}

/// The physical page atlas and the indirection buffer for one
/// [`inf_vsm::VsmResidency`].
pub struct VsmPools {
    atlas: wgpu::Texture,
    atlas_view: wgpu::TextureView,
    table: wgpu::Buffer,
    geometry: inf_vsm::VsmAtlasGeometry,
    /// The residency layout generation this buffer was built for. **A stamp, not
    /// a measurement** — the only legal operation on it is `!=`.
    layout_generation: u64,
    /// Bumped every time [`table`](Self::table) becomes a **different buffer**,
    /// so a cached bind group that names it can be dropped. `crate::vt`'s
    /// `table_generation`, for the hazard that field's docs describe.
    table_generation: u64,
}

impl VsmPools {
    /// Create the atlas and the indirection buffer for `residency`.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, residency: &VsmResidency) -> Self {
        let geometry = residency.geometry();
        // An atlas whose budget bought no slots still has to produce a legal
        // texture; it can hold nothing, so one empty slot costs nothing and keeps
        // the type total (`VtPools::new`'s shape).
        let extent = wgpu::Extent3d {
            width: geometry.atlas_width().max(geometry.stored_page_size),
            height: geometry.atlas_height().max(geometry.stored_page_size),
            depth_or_array_layers: 1,
        };
        let atlas = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vsm-page-atlas"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: VSM_PAGE_FORMAT,
            // RENDER_ATTACHMENT because P27.2 rasterizes into it; TEXTURE_BINDING
            // because P27.4 samples it. Nothing copies into it, ever.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let atlas_view = atlas.create_view(&wgpu::TextureViewDescriptor::default());
        let (table, _) = upload_whole_table(device, queue, residency);
        Self {
            atlas,
            atlas_view,
            table,
            geometry,
            layout_generation: residency.layout_generation(),
            table_generation: residency.layout_generation(),
        }
    }

    /// The atlas texture (for P27.2's per-page render passes).
    #[inline]
    pub fn atlas(&self) -> &wgpu::Texture {
        &self.atlas
    }
    #[inline]
    pub fn atlas_view(&self) -> &wgpu::TextureView {
        &self.atlas_view
    }
    /// The indirection buffer — every registered light's table, at the layout
    /// [`inf_vsm::table`] documents.
    #[inline]
    pub fn table(&self) -> &wgpu::Buffer {
        &self.table
    }
    #[inline]
    pub fn geometry(&self) -> inf_vsm::VsmAtlasGeometry {
        self.geometry
    }
    /// The generation of the **indirection buffer allocation** — it moves iff
    /// [`table`](Self::table) becomes a different buffer.
    #[inline]
    pub fn table_generation(&self) -> u64 {
        self.table_generation
    }
    /// VRAM the atlas costs.
    pub fn atlas_bytes(&self) -> u64 {
        u64::from(self.atlas.width())
            * u64::from(self.atlas.height())
            * inf_vsm::VSM_PAGE_BYTES_PER_TEXEL
    }

    /// **Apply a transaction, whole.**
    ///
    /// Table blocks are written for exactly the lights the transaction names,
    /// unless the layout moved — then the buffer is re-created and written
    /// whole, because a new light shifts every block after it.
    pub fn apply(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        residency: &VsmResidency,
        txn: &VsmTransaction,
    ) -> VsmApplyReport {
        let mut report = VsmApplyReport {
            pages: txn.admits.len(),
            evicted: txn.evicts.len(),
            ..Default::default()
        };
        if txn.layout_rebuilt || residency.layout_generation() != self.layout_generation {
            let (table, bytes) = upload_whole_table(device, queue, residency);
            self.table = table;
            self.layout_generation = residency.layout_generation();
            // A DIFFERENT allocation, so anything caching a bind group over it
            // has to be told. The residency's layout stamp is a global monotone
            // counter, so this never repeats a value a stale cache may hold.
            self.table_generation = residency.layout_generation();
            report.table_bytes = bytes;
            report.table_recreated = true;
        } else {
            for light in &txn.tables {
                let Some((base, words)) = residency.table_block(*light) else {
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
fn upload_whole_table(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    residency: &VsmResidency,
) -> (wgpu::Buffer, u64) {
    let words = residency.table_words();
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vsm-indirection"),
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

    /// The atlas format's texel size and `inf-vsm`'s page arithmetic cannot
    /// disagree without this failing — the crate that plans the budget and the
    /// crate that allocates the texture agree about what a page costs.
    #[test]
    fn the_page_format_agrees_with_the_page_size() {
        assert_eq!(
            u64::from(
                VSM_PAGE_FORMAT
                    .block_copy_size(Some(wgpu::TextureAspect::DepthOnly))
                    .expect("a depth aspect has a block size")
            ),
            inf_vsm::VSM_PAGE_BYTES_PER_TEXEL
        );
        assert_eq!(VSM_PAGE_FORMAT.block_dimensions(), (1, 1));
        assert_eq!(VSM_PAGE_FORMAT, crate::camera::DEPTH_FORMAT);
    }

    /// **The nineteen lines the separate-crate ruling copied still agree** — one
    /// door where a dependency edge is not allowed to be one.
    ///
    /// `inf-vsm`'s manifest and its crate docs both say why it does not name
    /// `inf-vt`: the two address spaces are different shapes, and the edge would
    /// buy one 30-line `pack_entry` at the price of coupling shadow paging to a
    /// texture container's tile geometry. The price it *does* pay is a copy, and
    /// a copy with no pin is a copy that drifts — which matters here because
    /// P28.3's whole plan is to merge the two brains, and a silent divergence in
    /// the entry format is a divergence in what the merged table means.
    ///
    /// This crate depends on **both**, so the pin can live here even though
    /// neither crate may hold it. It is a mechanical agreement check, not a
    /// re-derivation: `lru_victim` is `pub(crate)` in both and `slot_origin` sits
    /// on two different geometry types, so what is pinnable is the entry word and
    /// its slot ceiling — the two the wire format actually rests on.
    #[test]
    fn the_two_page_tables_pack_a_slot_the_same_way() {
        assert_eq!(
            inf_vsm::MAX_SLOT_INDEX,
            inf_vt::table::MAX_SLOT_INDEX,
            "the two crates disagree about how many slots an entry addresses"
        );
        for slot in [0u32, 1, 1_023, 2_713, inf_vsm::MAX_SLOT_INDEX] {
            for level in [0u32, 1, 8, 15] {
                assert_eq!(
                    inf_vsm::pack_entry(slot, level),
                    inf_vt::pack_entry(slot, level),
                    "slot {slot} level {level}: the copied `pack_entry` has drifted"
                );
            }
        }
        // …and the ONE deliberate difference is still deliberate: a shadow entry
        // has a NONE state a virtual texture's cannot have, which is why
        // `unpack_entry` was measured as *not* reusable.
        assert_eq!(inf_vsm::unpack_entry(inf_vsm::VSM_ENTRY_NONE), None);
        assert_eq!(
            inf_vt::unpack_entry(inf_vsm::VSM_ENTRY_NONE).slot,
            inf_vsm::MAX_SLOT_INDEX,
            "`inf-vt`'s unpack has no absence to return, which is the measurement"
        );
    }
}
