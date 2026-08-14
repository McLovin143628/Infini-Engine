//! The `.inf_vmesh` **streaming asset** (P18.2): the meshlet DAG leaves its
//! single bincode blob for a random-access, page-at-a-time container.
//!
//! ```text
//! ┌ header (128 B, little-endian) ────────────────────────────────────┐
//! │  magic         [u8; 8]   b"INFVMSH\0"                             │
//! │  schema_ver    u32       VMESH_ASSET_SCHEMA_VERSION (3)           │
//! │  page_count    u32       streaming pages (coarsest/root page 0)   │
//! │  meshlet_count u32       meshlets across all pages                │
//! │  vertex_count  u32       vertex records across all pages          │
//! │  max_tri       u32       max triangle_count (the indirect draw's) │
//! │  max_lod       u32       coarsest LOD level present               │
//! │  center        [f32; 3] · radius f32                              │
//! │  page_base     u64       absolute offset of the section area      │
//! │  groups_off    u64 · groups_len u64   the DAG groups (bincode)    │
//! │  total_len     u64       payload length (a self-check)            │
//! │  reserved      [u8; 40]  zeros (room for v4 without a re-length)  │
//! ├ page directory (page_count × 96 B, COARSEST FIRST) ───────────────┤
//! │  page u32 · meshlet_count u32 · vertex_start u32 · vertex_count   │
//! │  u32 · mlvert_count u32 · mltri_count u32 · floor_lod u32 ·       │
//! │  lod u32 · max_parent_error f32 · min_error f32 · tile_count u32  │
//! │  · pad u32 · indices_off · meshlets_off · mlverts_off ·           │
//! │  mltris_off · vertices_off (u64 each) · tiles_off u64             │
//! ├ page sections, INTERLEAVED per cluster page (v3) ─────────────────┤
//! │  page 0: indices · meshlets · mlverts · mltris · vertices · TILES │
//! │  page 1: indices · meshlets · mlverts · mltris · vertices · TILES │
//! │  …  each 16-byte aligned, zero-padded up to the next boundary …   │
//! ├ groups section (bincode `Vec<Group>`) ────────────────────────────┤
//! └───────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # v3: a cluster page and its texture tiles are one page (P28.2)
//!
//! The **tiles section** is what makes a cluster page carry its texture. It is a
//! run of [`ClusterTileRef`] records — the virtual-texture tiles the page's
//! materials sample **at that page's detail level** — laid down immediately
//! after the page's own vertex section and before the next page's, so one
//! cluster page's geometry and its texture addresses are contiguous bytes rather
//! than two systems that have to be kept in step.
//!
//! **The tiles section holds addresses, not texels, and that was decided by
//! measurement** — see `docs/memos/p28-2-cluster-pages.md`. A `.inf_tex` v2
//! container is already a byte-addressable, uncompressed, 16-byte-aligned tile
//! image inside the *same* mmap'd pack (P26.1), so copying a tile's bytes in
//! here would duplicate what the mapping already hands back as a borrowed slice,
//! at a measured multiple, and buy no I/O: two slices of one mapping are one
//! read. What the interleaving buys is the **pairing** — the page directory
//! names the page's tiles, so one page-in transaction can feed both consumers
//! and neither can be admitted without the other.
//!
//! v2 and v1 both keep loading, by lifting: see [`VgeomSource::from_payload`].
//!
//! # Why a sectioned v2 and not a range-read of v1
//!
//! v1 was `inf_asset::encode(&VgeomMesh)` — one bincode stream. Every `Vec` field
//! is a length prefix followed by a run of **varint-packed** records, so there is
//! no byte offset for "level 3's meshlets" that does not require decoding
//! everything before it. The level-major layout [`VgeomMesh`] already guarantees
//! makes the *logical* ranges contiguous; v2 is what turns those into **byte**
//! ranges. Same shape, and the same reasoning, as the `.inf_terrain` container
//! (P16.3): header + sorted directory + 16-byte-aligned blobs, cooked
//! uncompressed so a packed entry is a borrowed slice of the mmap.
//!
//! **v1 keeps loading forever.** [`VgeomSource::from_payload`] sniffs the magic; a
//! payload without it is decoded as a v1 `VgeomMesh` and re-laid-out in memory, so
//! a pack cooked before this batch still runs (slower to open, identical
//! afterwards). A bincode `VgeomMesh` begins with `schema_version: u32 = 1`
//! (`01 00 00 00`), which cannot collide with the magic.
//!
//! # Pages, not LOD levels — and why that is the never-a-hole guarantee
//!
//! The obvious paging unit is the LOD level, and it is **wrong**. A group that
//! fails to simplify leaves its members as *roots* (`parent_error == +∞`) at
//! whatever level they reached (`build.rs`: `if !res.progressed { continue }`), so
//! roots live at **many** levels, not just the coarsest. Evicting "everything
//! finer than level F" would evict a level-2 root whose path has nothing coarser
//! at all — a hole, exactly the failure this design is required to make
//! unreachable.
//!
//! So the paging unit is a **page**:
//!
//! * **page 0** — *every root, from every level*. Always resident. It is the
//!   complete coarsest cut: at a large enough threshold the DAG cut selects
//!   exactly the roots, so page 0 alone can draw the whole mesh.
//! * **page p ≥ 1** — the **non-root** meshlets at LOD level `max_lod - p`,
//!   coarse to fine. A page with no non-roots is skipped entirely.
//!
//! Residency is a **prefix** of that order, which makes the two properties the
//! cut clamp needs true by construction:
//!
//! 1. *Ancestor closure* — a non-root at level `L` has its parent at `L+1`, which
//!    is either a root (page 0) or a non-root at `L+1` (an earlier page).
//! 2. *Never a hole* — every root-to-leaf path ends at a root, and page 0 is never
//!    evicted, so **every path always has at least one resident meshlet**.
//!
//! [`VgeomMesh::select_with_residency`] is the CPU reference for the resulting
//! clamped cut, and `vgeom_cull.wgsl` applies the identical rule.
//!
//! # Vertex blocks: stored once, resident as a prefix
//!
//! `VgeomMesh::vertices` is one shared, welded buffer — a coarse meshlet's
//! vertices are a *subset* of the finer ones' — so naive per-page vertex blocks
//! would store a vertex once per page that touches it. Instead the image
//! **permutes** the vertex buffer into page order: `page(v)` is the coarsest page
//! that references `v`, vertices are emitted in ascending `page(v)`, and a page's
//! block is the *increment*. Every vertex is stored exactly once, page `p`'s
//! meshlets reference only `[0, prefix(p))`, and a prefix residency therefore
//! holds a complete, contiguous vertex prefix. The permutation is internal to the
//! container: `mlverts` are stored in the image's numbering and
//! [`VgeomAssetReader::to_mesh`] hands back a mesh whose vertices are permuted the
//! same way — geometrically identical, meshlet indices and micro-triangle bytes
//! untouched.
//!
//! # There is exactly one writer: [`VgeomAssetImage::as_bytes`]
//!
//! Like `.inf_terrain`, the bytes on disk and in a `.inf_pack` are the **raw
//! image**, never `inf_asset::encode` output — a bincode length prefix would shift
//! every section off its 16-byte boundary and defeat the whole layout, silently.
//! So [`VgeomAssetImage`] deliberately implements neither `AssetPayload` nor
//! `Serialize`, which makes the generic asset-writing doors a type error rather
//! than a subtly wrong file.

use std::borrow::Cow;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use inf_asset::{AssetId, AssetKind, PackReader};

use crate::model::{Group, LevelRange, Meshlet, VgeomMesh, VgeomVertex};

/// Magic at the head of every v2 `.inf_vmesh` payload.
pub const VMESH_ASSET_MAGIC: [u8; 8] = *b"INFVMSH\0";

/// Current `.inf_vmesh` **container** schema version.
///
/// v1 is the bare bincode [`VgeomMesh`] (no magic); v2 is the paged image P18.2
/// wrote; v3 adds the per-page **tiles** section and widens the vertex record
/// with the tangent channel. All three keep loading — v1 and v2 by lifting into
/// a v3 image at open (see [`VgeomSource::from_payload`]), which is the same
/// arrangement, and the same cost note, P18.2 gave v1.
pub const VMESH_ASSET_SCHEMA_VERSION: u32 = 3;

/// Sections start on multiples of this many bytes — the same constant, and the
/// same reasoning, as [`inf_asset::BLOB_ALIGN`] and `.inf_terrain`'s `TILE_ALIGN`.
pub const SECTION_ALIGN: u64 = 16;

/// Bytes of the fixed header.
pub const HEADER_LEN: u64 = 128;

/// Bytes of one page-directory entry.
pub const PAGE_ENTRY_LEN: u64 = 96;

/// Bytes of one on-disk meshlet record ([`MeshletRec`]).
pub const MESHLET_REC_LEN: usize = 64;

/// Bytes of one on-disk vertex record ([`VgeomVertex`]) — **36 since v3**, which
/// is where the tangent word joined position + normal + uv.
pub const VERTEX_REC_LEN: usize = 36;

/// Bytes of a **v2** vertex record: position + normal + uv, no tangent. Read
/// only by the v2 lift, which is the one place two record widths meet.
pub const VERTEX_REC_LEN_V2: usize = 32;

/// Bytes of one on-disk cluster tile reference ([`ClusterTileRef`]).
pub const TILE_REF_LEN: usize = 32;

/// Bytes of one stored vertex record at container version `schema`.
#[inline]
pub const fn vertex_rec_len(schema: u32) -> usize {
    if schema >= 3 {
        VERTEX_REC_LEN
    } else {
        VERTEX_REC_LEN_V2
    }
}

/// A failure building or reading a `.inf_vmesh` payload.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum VgeomAssetError {
    #[error("payload is shorter than the fixed header")]
    TooShort,
    #[error("bad .inf_vmesh magic")]
    BadMagic,
    #[error("payload schema v{found} is newer than this build's v{current}")]
    SchemaTooNew { found: u32, current: u32 },
    #[error("malformed .inf_vmesh payload: {0}")]
    Malformed(String),
    #[error("page {page} section `{section}` is out of bounds or misaligned")]
    SectionOutOfBounds { page: usize, section: &'static str },
    #[error("groups section failed to decode: {0}")]
    GroupsDecode(String),
    #[error("page {page} meshlet {index}: {what} range is outside its section")]
    RecordOutOfBounds {
        page: usize,
        index: usize,
        what: &'static str,
    },
}

type Result<T> = std::result::Result<T, VgeomAssetError>;

/// Round `n` up to the next multiple of [`SECTION_ALIGN`].
#[inline]
const fn align_up(n: u64) -> u64 {
    n.next_multiple_of(SECTION_ALIGN)
}

/// One meshlet **as it is stored and as the GPU reads it** — 64 bytes, the exact
/// layout of `MeshletGpu` / `struct Meshlet` in `vgeom_cull.wgsl`.
///
/// Storing the GPU record makes paging a page in a memcpy plus two `u32` rebases
/// (`vertex_offset` / `triangle_offset` are **page-local** on disk and
/// pool-absolute in VRAM) rather than a decode. `group` occupies the word the GPU
/// struct pads; the shaders never read it, and materialization needs it.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct MeshletRec {
    pub center: [f32; 3],
    pub radius: f32,
    pub cone_axis: [f32; 3],
    pub cone_cutoff: f32,
    /// Offset into **this page's** `mlverts` section (records, not bytes).
    pub vertex_offset: u32,
    /// Byte offset into **this page's** `mltris` section.
    pub triangle_offset: u32,
    pub vertex_count: u32,
    pub triangle_count: u32,
    pub error: f32,
    pub parent_error: f32,
    pub lod_level: u32,
    /// [`Meshlet::group`] — inert on the GPU (it is the record's pad word).
    pub group: u32,
}

const _: () = assert!(std::mem::size_of::<MeshletRec>() == MESHLET_REC_LEN);
const _: () = assert!(std::mem::size_of::<VgeomVertex>() == VERTEX_REC_LEN);
const _: () = assert!(std::mem::size_of::<ClusterTileRef>() == TILE_REF_LEN);

/// One virtual-texture tile a cluster page's materials sample — the record the
/// v3 **tiles** section is a run of, and the whole pairing contract in 32 bytes.
///
/// The texture id is stored as its two `u64` halves rather than as an
/// `inf_asset::AssetId`, because this is a `Pod` record sliced straight out of an
/// mmap and a `Uuid`'s representation is not this module's to freeze.
/// [`ClusterTileRef::texture`] is the one conversion, and
/// [`ClusterTileRef::coord`] is the other — a `(mip, x, y)` triple in
/// `inf_vt`'s own spelling, so a tile address has one name in the tree.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Pod, Zeroable)]
pub struct ClusterTileRef {
    /// Low 64 bits of the `.inf_tex` asset GUID.
    pub texture_lo: u64,
    /// High 64 bits of the `.inf_tex` asset GUID.
    pub texture_hi: u64,
    pub mip: u32,
    pub x: u32,
    pub y: u32,
    /// Zero. Present so the record is 32 bytes — two to a 16-byte lane, which is
    /// what keeps the section's alignment arithmetic exact.
    pub pad: u32,
}

impl ClusterTileRef {
    pub fn new(texture: AssetId, mip: u32, x: u32, y: u32) -> Self {
        let v = texture.uuid().as_u128();
        Self {
            texture_lo: v as u64,
            texture_hi: (v >> 64) as u64,
            mip,
            x,
            y,
            pad: 0,
        }
    }

    /// The `.inf_tex` asset this tile belongs to.
    pub fn texture(&self) -> AssetId {
        AssetId(uuid::Uuid::from_u128(
            (u128::from(self.texture_hi) << 64) | u128::from(self.texture_lo),
        ))
    }

    /// The tile address, in `inf_vt`'s spelling.
    pub fn coord(&self) -> inf_vt::TileCoord {
        inf_vt::TileCoord::new(self.mip, self.x, self.y)
    }

    /// The order the section is sorted in: by texture, then by the order the
    /// `.inf_tex` tile directory itself stores tiles (`(mip, y, x)` — the P26.1
    /// audit's finding), so walking a page's tiles is one forward scan of each
    /// texture's mapping rather than a scatter.
    fn sort_key(&self) -> (u64, u64, u32, u32, u32) {
        (self.texture_hi, self.texture_lo, self.mip, self.y, self.x)
    }
}

/// One texture a mesh's materials reference, as the pairing needs to see it.
///
/// Deliberately not a `.inf_tex` reader: the cook has already opened the
/// container to learn these four numbers, and a builder that took a reader would
/// make `inf-vgeom` depend on the texture *file* rather than on the address
/// space. `inf_vt::VtTextureDesc` is where these come from
/// (`TiledTextureReader::vt_desc`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClusterTexture {
    pub id: AssetId,
    /// Tiles across / down at each mip, **finest first** — exactly the
    /// container's mip directory.
    pub mips: Vec<(u32, u32)>,
}

impl ClusterTexture {
    /// From a virtual-texture descriptor — the only producer outside tests.
    pub fn from_desc(id: AssetId, desc: &inf_vt::VtTextureDesc) -> Self {
        Self {
            id,
            mips: desc.mips.iter().map(|m| (m.tiles_x, m.tiles_y)).collect(),
        }
    }
}

/// The textures a `.inf_vmesh`'s clusters are paired against at cook.
///
/// [`ClusterTextureSet::none`] is the honest default and it is what every fixture
/// and every un-textured mesh builds with: no pairing, no tiles section, and the
/// runtime coupling is inert. A mesh acquires a pairing only when the cook can
/// see its materials (the P26.3b wire).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClusterTextureSet {
    /// In `inf_asset::DerivedMaterial::texture_dependencies` order (albedo →
    /// normal → ORM), deduped — the order that is already the residency
    /// contract, reused rather than re-invented.
    pub textures: Vec<ClusterTexture>,
}

impl ClusterTextureSet {
    /// No pairing. The v3 image is then byte-for-byte what v3 without this
    /// feature would be, and every page's `tile_count` is zero.
    pub fn none() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.textures.is_empty()
    }
}

/// **The cluster → tile rule**, stated once and read by the builder, the cook and
/// the gate's independent oracle.
///
/// A LOD level halves a page's triangle count; a mip level **quarters** its
/// texel count. So one mip is worth two LOD levels, and the pairing steps the
/// texture down half as fast as the geometry:
///
/// ```text
///     mip(level L) = min(mip_count - 1, L / 2)          (integer division)
///     mip(root page) = mip_count - 1                    (the coarsest level)
/// ```
///
/// The finest geometry therefore pairs with **mip 0** — which is the whole
/// point, because the artifact this phase makes impossible is a high-poly mesh
/// with a blurry texture — and the always-resident root page pairs with the
/// always-resident coarsest mip, so its pairing is a no-op against the virtual
/// texture's own mandatory floor.
///
/// The alternative, measured and not taken, is a **density** rule (pick the
/// finest mip whose texels-per-triangle stays under a constant). It is stated in
/// `docs/memos/p28-2-cluster-pages.md` with the number that rejected it: on the
/// 96² fixture against a 2 048² texture it caps the finest page at mip 1, so the
/// finest texture level is never paired at all and the artifact survives at
/// close range.
pub fn tile_mip_for_lod(lod: u32, mip_count: u32) -> u32 {
    let coarsest = mip_count.saturating_sub(1);
    if lod == u32::MAX {
        // The root page spans every level; it pairs with the coarsest mip.
        return coarsest;
    }
    (lod / 2).min(coarsest)
}

/// The decoded fixed header of a `.inf_vmesh` payload.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VgeomAssetHeader {
    pub schema_version: u32,
    /// Streaming pages present (the directory length).
    pub page_count: u32,
    /// Meshlets across every page.
    pub meshlet_count: u32,
    /// Vertex records across every page.
    pub vertex_count: u32,
    /// Largest `triangle_count` over every meshlet — the vertex-pulled indirect
    /// draw's fixed `vertex_count / 3`. Read from the header, so the draw shape is
    /// known **without paging anything in**.
    pub max_tri: u32,
    /// Coarsest LOD level present.
    pub max_lod: u32,
    /// Whole-mesh bounding-sphere centre (local space).
    pub center: [f32; 3],
    /// Whole-mesh bounding-sphere radius.
    pub radius: f32,
    /// Absolute offset of the first page section.
    pub page_base: u64,
    /// The DAG groups section (bincode `Vec<Group>`).
    pub groups_off: u64,
    pub groups_len: u64,
    /// Payload length as written.
    pub total_len: u64,
}

/// One page-directory entry: everything the streamer needs about a page
/// **before** any of its bytes are touched.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VgeomPageEntry {
    /// Page index (0 = the always-resident root page).
    pub page: u32,
    /// Meshlets in this page.
    pub meshlet_count: u32,
    /// First image-vertex index of this page's incremental vertex block.
    pub vertex_start: u32,
    /// Vertices this page is the coarsest consumer of (see the module docs).
    pub vertex_count: u32,
    /// Micro vertex-index entries (`u32`s).
    pub mlvert_count: u32,
    /// Micro triangle-index bytes.
    pub mltri_count: u32,
    /// The LOD level at or below which a meshlet's error is treated as **0** when
    /// this page is the finest resident one — the residency clamp's only
    /// parameter (see [`VgeomMesh::select_with_residency`]). `max_lod` for the
    /// root page; the page's own level otherwise.
    pub floor_lod: u32,
    /// The LOD level this page's meshlets sit at (`u32::MAX` for the root page,
    /// which spans every level).
    pub lod: u32,
    /// Largest `parent_error` over this page's meshlets — the **want key**: this
    /// page can contribute to the cut at threshold `t` only while
    /// `max_parent_error > t`. `+∞` for the root page, which is always wanted.
    pub max_parent_error: f32,
    /// Smallest `error` over this page's meshlets (diagnostics + load priority).
    pub min_error: f32,
    /// [`ClusterTileRef`] records in this page's **tiles** section (v3). Zero for
    /// a v2 image and for any mesh cooked without a material pairing.
    pub tile_count: u32,
    /// `u32` global meshlet index per meshlet — how the remap table is built.
    pub indices_off: u64,
    pub meshlets_off: u64,
    pub mlverts_off: u64,
    pub mltris_off: u64,
    pub vertices_off: u64,
    /// Absolute offset of this page's tiles section (v3). Zero when
    /// [`tile_count`](Self::tile_count) is zero.
    pub tiles_off: u64,
}

impl VgeomPageEntry {
    /// Bytes this page occupies in the four GPU pools if it is made resident —
    /// the quantity the VRAM budget is spent in. The `indices` section is CPU-side
    /// bookkeeping and is deliberately **not** counted: it is never uploaded, and
    /// neither is the v3 `tiles` section, which is a list of **addresses**: the
    /// texels those addresses name are spent out of the virtual texture's own
    /// page budget, by the same transaction, and counting them here would spend
    /// them twice.
    pub fn resident_bytes(&self) -> u64 {
        self.meshlet_count as u64 * MESHLET_REC_LEN as u64
            + self.vertex_count as u64 * VERTEX_REC_LEN as u64
            + self.mlvert_count as u64 * 4
            + u64::from(self.mltri_count).next_multiple_of(4)
    }

    /// Whether this is the always-resident root page.
    #[inline]
    pub fn is_root_page(&self) -> bool {
        self.page == 0
    }
}

/// The borrowed byte sections of one page.
#[derive(Debug, Clone, Copy)]
pub struct VgeomPageSections<'a> {
    /// `meshlet_count` × 4 bytes: the global meshlet index of each record.
    pub indices: &'a [u8],
    /// `meshlet_count` × [`MESHLET_REC_LEN`] bytes of [`MeshletRec`].
    pub meshlets: &'a [u8],
    /// `mlvert_count` × 4 bytes of **image-numbered** vertex indices.
    pub mlverts: &'a [u8],
    /// `mltri_count` bytes of packed micro triangle indices.
    pub mltris: &'a [u8],
    /// `vertex_count` × [`VERTEX_REC_LEN`] bytes of [`VgeomVertex`].
    pub vertices: &'a [u8],
    /// `tile_count` × [`TILE_REF_LEN`] bytes of [`ClusterTileRef`] — the tiles
    /// this cluster page's materials sample at its detail level (v3).
    pub tiles: &'a [u8],
}

impl VgeomPageSections<'_> {
    /// This page's tile references, as records.
    ///
    /// The cast is exact: the section is 16-byte aligned inside a 16-byte-aligned
    /// payload and its length was checked to be a whole multiple of
    /// [`TILE_REF_LEN`] at parse.
    pub fn tile_refs(&self) -> &[ClusterTileRef] {
        bytemuck::cast_slice(self.tiles)
    }
}

// ── builder ─────────────────────────────────────────────────────────────────

/// Lay a [`VgeomMesh`] out as a v3 paged image, pairing every cluster page with
/// the texture tiles its materials sample at that page's detail level.
///
/// Pure and byte-deterministic: the page partition, the vertex permutation, the
/// section order **and the pairing** are all functions of `(mesh, textures)`
/// alone, and padding is deterministic zeros — so two builds of one mesh against
/// one texture set are byte-identical (the cook's guarantee).
///
/// Pass [`ClusterTextureSet::none`] for a mesh with no material pairing; every
/// page then carries an empty tiles section and the runtime coupling is inert.
pub fn build_vgeom_asset(
    mesh: &VgeomMesh,
    textures: &ClusterTextureSet,
) -> Result<VgeomAssetImage> {
    validate(mesh)?;
    let max_lod = mesh.meshlets.iter().map(|m| m.lod_level).max().unwrap_or(0) as u32;

    // ── 1. Partition meshlets into pages ──
    // Page 0 = every root (from every level). Page p ≥ 1 = the NON-root meshlets
    // at level `max_lod - p`. Empty pages are skipped. Within a page, meshlets
    // keep ascending global index order (deterministic).
    let mut page_members: Vec<Vec<u32>> = Vec::new();
    let mut roots: Vec<u32> = Vec::new();
    let mut by_lod: Vec<Vec<u32>> = vec![Vec::new(); max_lod as usize + 1];
    for (i, m) in mesh.meshlets.iter().enumerate() {
        if m.is_root() {
            roots.push(i as u32);
        } else {
            by_lod[m.lod_level as usize].push(i as u32);
        }
    }
    let mut page_lod: Vec<u32> = Vec::new();
    if !roots.is_empty() {
        page_members.push(roots);
        page_lod.push(u32::MAX);
    }
    for lod in (0..=max_lod).rev() {
        let members = std::mem::take(&mut by_lod[lod as usize]);
        if members.is_empty() {
            continue;
        }
        page_members.push(members);
        page_lod.push(lod);
    }
    let page_count = page_members.len();

    // ── 2. Permute vertices into page order ──
    // `page_of[v]` = the coarsest page referencing `v`; unreferenced vertices sort
    // last and are never emitted (nothing can draw them).
    const UNREFERENCED: u32 = u32::MAX;
    let mut page_of = vec![UNREFERENCED; mesh.vertices.len()];
    for (p, members) in page_members.iter().enumerate() {
        for &mi in members {
            let m = &mesh.meshlets[mi as usize];
            let lo = m.vertex_offset as usize;
            let hi = lo + m.vertex_count as usize;
            for &v in &mesh.meshlet_vertices[lo..hi] {
                let slot = &mut page_of[v as usize];
                *slot = (*slot).min(p as u32);
            }
        }
    }
    // Stable ordering: ascending page, then ascending original index.
    let mut order: Vec<u32> = (0..mesh.vertices.len() as u32).collect();
    order.sort_by_key(|&v| (page_of[v as usize], v));
    let mut image_index = vec![0u32; mesh.vertices.len()];
    for (new, &old) in order.iter().enumerate() {
        image_index[old as usize] = new as u32;
    }
    // Per-page vertex block = the increment of the referenced prefix.
    let mut page_vertex_start = vec![0u32; page_count];
    let mut page_vertex_count = vec![0u32; page_count];
    {
        let mut cursor = 0u32;
        for (p, (start, count)) in page_vertex_start
            .iter_mut()
            .zip(page_vertex_count.iter_mut())
            .enumerate()
        {
            let n = order[cursor as usize..]
                .iter()
                .take_while(|&&v| page_of[v as usize] == p as u32)
                .count() as u32;
            *start = cursor;
            *count = n;
            cursor += n;
        }
    }

    // ── 3. Gather each page's sections ──
    struct PageBlob {
        indices: Vec<u8>,
        meshlets: Vec<u8>,
        mlverts: Vec<u8>,
        mltris: Vec<u8>,
        tiles: Vec<u8>,
        max_parent_error: f32,
        min_error: f32,
    }
    let mut blobs: Vec<PageBlob> = Vec::with_capacity(page_count);
    for (p, members) in page_members.iter().enumerate() {
        let mut indices = Vec::with_capacity(members.len() * 4);
        let mut meshlets = Vec::with_capacity(members.len() * MESHLET_REC_LEN);
        let mut mlverts: Vec<u32> = Vec::new();
        let mut mltris: Vec<u8> = Vec::new();
        let mut max_parent_error = 0.0f32;
        let mut min_error = f32::INFINITY;
        for &mi in members {
            let m = &mesh.meshlets[mi as usize];
            indices.extend_from_slice(&mi.to_le_bytes());
            let rec = MeshletRec {
                center: m.center,
                radius: m.radius,
                cone_axis: m.cone_axis,
                cone_cutoff: m.cone_cutoff,
                vertex_offset: mlverts.len() as u32,
                triangle_offset: mltris.len() as u32,
                vertex_count: m.vertex_count,
                triangle_count: m.triangle_count,
                error: m.error,
                parent_error: m.parent_error,
                lod_level: m.lod_level as u32,
                group: m.group,
            };
            meshlets.extend_from_slice(bytemuck::bytes_of(&rec));
            let lo = m.vertex_offset as usize;
            mlverts.extend(
                mesh.meshlet_vertices[lo..lo + m.vertex_count as usize]
                    .iter()
                    .map(|&v| image_index[v as usize]),
            );
            let lo = m.triangle_offset as usize;
            mltris
                .extend_from_slice(&mesh.meshlet_triangles[lo..lo + m.triangle_count as usize * 3]);
            max_parent_error = max_parent_error.max(m.parent_error);
            min_error = min_error.min(m.error);
        }
        let refs = pair_page_tiles(mesh, members, page_lod[p], textures);
        blobs.push(PageBlob {
            indices,
            meshlets,
            mlverts: bytemuck::cast_slice(&mlverts).to_vec(),
            mltris,
            tiles: bytemuck::cast_slice(&refs).to_vec(),
            max_parent_error,
            min_error: if min_error.is_finite() {
                min_error
            } else {
                0.0
            },
        });
    }

    // ── 4. Offsets ──
    let dir_len = PAGE_ENTRY_LEN * page_count as u64;
    let page_base = align_up(HEADER_LEN + dir_len);
    let mut off = page_base;
    // The six sections of a page are laid down together, page after page — which
    // is what "interleaved per virtual cluster page" is, in bytes: a page's tile
    // addresses sit between its own vertices and the next page's indices.
    let mut offsets: Vec<[u64; 6]> = Vec::with_capacity(page_count);
    for (p, b) in blobs.iter().enumerate() {
        let mut here = [0u64; 6];
        for (slot, len) in [
            b.indices.len() as u64,
            b.meshlets.len() as u64,
            b.mlverts.len() as u64,
            b.mltris.len() as u64,
            page_vertex_count[p] as u64 * VERTEX_REC_LEN as u64,
            b.tiles.len() as u64,
        ]
        .into_iter()
        .enumerate()
        {
            here[slot] = off;
            off = align_up(off + len);
        }
        offsets.push(here);
    }
    let groups_off = off;
    let groups_bytes = bincode::serde::encode_to_vec(&mesh.groups, inf_asset::bincode_config())
        .map_err(|e| VgeomAssetError::Malformed(format!("encode groups: {e}")))?;
    let total_len = align_up(groups_off + groups_bytes.len() as u64);

    let max_tri = mesh
        .meshlets
        .iter()
        .map(|m| m.triangle_count)
        .max()
        .unwrap_or(0);

    // ── 5. Emit ──
    let mut out: Vec<u8> = Vec::with_capacity(total_len as usize);
    out.extend_from_slice(&VMESH_ASSET_MAGIC);
    out.extend_from_slice(&VMESH_ASSET_SCHEMA_VERSION.to_le_bytes());
    out.extend_from_slice(&(page_count as u32).to_le_bytes());
    out.extend_from_slice(&(mesh.meshlets.len() as u32).to_le_bytes());
    out.extend_from_slice(&(mesh.vertices.len() as u32).to_le_bytes());
    out.extend_from_slice(&max_tri.to_le_bytes());
    out.extend_from_slice(&max_lod.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // pad to 16-byte lane
    for v in mesh.center {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&mesh.radius.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // pad: the u64 lane starts at 56
    debug_assert_eq!(out.len(), 56);
    out.extend_from_slice(&page_base.to_le_bytes());
    out.extend_from_slice(&groups_off.to_le_bytes());
    out.extend_from_slice(&(groups_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&total_len.to_le_bytes());
    debug_assert!(out.len() as u64 <= HEADER_LEN);
    out.resize(HEADER_LEN as usize, 0);

    for p in 0..page_count {
        let b = &blobs[p];
        let lod = page_lod[p];
        // The clamp floor when THIS page is the finest resident one: the root page
        // leaves only the roots, which behave as level `max_lod`; a level page
        // makes its own level the floor.
        let floor_lod = if lod == u32::MAX { max_lod } else { lod };
        out.extend_from_slice(&(p as u32).to_le_bytes());
        out.extend_from_slice(&(page_members[p].len() as u32).to_le_bytes());
        out.extend_from_slice(&page_vertex_start[p].to_le_bytes());
        out.extend_from_slice(&page_vertex_count[p].to_le_bytes());
        out.extend_from_slice(&((b.mlverts.len() / 4) as u32).to_le_bytes());
        out.extend_from_slice(&(b.mltris.len() as u32).to_le_bytes());
        out.extend_from_slice(&floor_lod.to_le_bytes());
        out.extend_from_slice(&lod.to_le_bytes());
        out.extend_from_slice(&b.max_parent_error.to_le_bytes());
        out.extend_from_slice(&b.min_error.to_le_bytes());
        out.extend_from_slice(&((b.tiles.len() / TILE_REF_LEN) as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // pad
        for v in offsets[p] {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    debug_assert_eq!(out.len() as u64, HEADER_LEN + dir_len);
    out.resize(page_base as usize, 0);

    for p in 0..page_count {
        let b = &blobs[p];
        let o = &offsets[p];
        for (slot, bytes) in [
            b.indices.as_slice(),
            b.meshlets.as_slice(),
            b.mlverts.as_slice(),
            b.mltris.as_slice(),
        ]
        .into_iter()
        .enumerate()
        {
            out.resize(o[slot] as usize, 0);
            out.extend_from_slice(bytes);
        }
        out.resize(o[4] as usize, 0);
        let start = page_vertex_start[p] as usize;
        for &old in &order[start..start + page_vertex_count[p] as usize] {
            out.extend_from_slice(bytemuck::bytes_of(&mesh.vertices[old as usize]));
        }
        out.resize(o[5] as usize, 0);
        out.extend_from_slice(&b.tiles);
    }

    out.resize(groups_off as usize, 0);
    out.extend_from_slice(&groups_bytes);
    out.resize(total_len as usize, 0);

    Ok(VgeomAssetImage { bytes: out })
}

/// **The pairing**: which texture tiles one cluster page's materials sample.
///
/// Two inputs and nothing else, so it is a pure function of the cook's own data:
///
/// 1. the page's **uv footprint** — the axis-aligned bound of every uv the
///    page's meshlets reference. Conservative on purpose: a bound can only ever
///    name tiles a triangle does not touch, never miss one it does, and the
///    invariant this pairing exists to make true is a *superset* claim.
/// 2. the page's **detail level** → a mip, by [`tile_mip_for_lod`].
///
/// The result is sorted into payload order and deduped, so the section is one
/// forward scan of each texture's mapping and two builds are byte-identical.
///
/// A uv outside `[0, 1]` is **wrapped** rather than clamped, because that is what
/// a sampler does with the default address mode and a tiled uv is ordinary
/// authored content. A non-finite uv contributes nothing — a NaN bound would
/// otherwise swallow the whole grid through `as u32`'s saturating cast, which is
/// the quiet version of "this page wants every tile there is".
fn pair_page_tiles(
    mesh: &VgeomMesh,
    members: &[u32],
    lod: u32,
    textures: &ClusterTextureSet,
) -> Vec<ClusterTileRef> {
    if textures.is_empty() || members.is_empty() {
        return Vec::new();
    }
    // The page's uv footprint, in wrapped uv space.
    let (mut u0, mut u1, mut v0, mut v1) = (f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY);
    let mut any = false;
    for &mi in members {
        let m = &mesh.meshlets[mi as usize];
        let lo = m.vertex_offset as usize;
        for &vi in &mesh.meshlet_vertices[lo..lo + m.vertex_count as usize] {
            let uv = mesh.vertices[vi as usize].uv;
            if !uv[0].is_finite() || !uv[1].is_finite() {
                continue;
            }
            let (u, v) = (uv[0].rem_euclid(1.0), uv[1].rem_euclid(1.0));
            u0 = u0.min(u);
            u1 = u1.max(u);
            v0 = v0.min(v);
            v1 = v1.max(v);
            any = true;
        }
    }
    if !any {
        return Vec::new();
    }

    let mut out: Vec<ClusterTileRef> = Vec::new();
    for tex in &textures.textures {
        let mip = tile_mip_for_lod(lod, tex.mips.len() as u32);
        let Some(&(tiles_x, tiles_y)) = tex.mips.get(mip as usize) else {
            continue;
        };
        if tiles_x == 0 || tiles_y == 0 {
            continue;
        }
        // `min(.., last)` rather than a modulo: uv == 1.0 is the far edge of the
        // last tile, not the first tile of a second copy.
        let tile_of = |c: f32, n: u32| -> u32 { ((c * n as f32) as u32).min(n - 1) };
        let (tx0, tx1) = (tile_of(u0, tiles_x), tile_of(u1, tiles_x));
        let (ty0, ty1) = (tile_of(v0, tiles_y), tile_of(v1, tiles_y));
        for y in ty0..=ty1 {
            for x in tx0..=tx1 {
                out.push(ClusterTileRef::new(tex.id, mip, x, y));
            }
        }
    }
    out.sort_by_key(ClusterTileRef::sort_key);
    out.dedup();
    out
}

/// Reject a mesh whose micro-index ranges do not fit its buffers, so every later
/// slice in this module is unchecked-safe.
fn validate(mesh: &VgeomMesh) -> Result<()> {
    for (i, m) in mesh.meshlets.iter().enumerate() {
        let v_end = m.vertex_offset as usize + m.vertex_count as usize;
        let t_end = m.triangle_offset as usize + m.triangle_count as usize * 3;
        if v_end > mesh.meshlet_vertices.len() || t_end > mesh.meshlet_triangles.len() {
            return Err(VgeomAssetError::Malformed(format!(
                "meshlet {i} micro-index range is out of bounds"
            )));
        }
        if mesh.meshlet_vertices[m.vertex_offset as usize..v_end]
            .iter()
            .any(|&v| v as usize >= mesh.vertices.len())
        {
            return Err(VgeomAssetError::Malformed(format!(
                "meshlet {i} references a vertex past the buffer"
            )));
        }
    }
    Ok(())
}

/// A validated v2 `.inf_vmesh` payload image. Owns its bytes; this — not
/// `inf_asset::encode` — is what goes on disk and into a pack.
#[derive(Clone, PartialEq)]
pub struct VgeomAssetImage {
    bytes: Vec<u8>,
}

impl VgeomAssetImage {
    /// The asset kind + container schema version this image owes the database (it
    /// implements no `AssetPayload` on purpose — see the module docs).
    pub const KIND: AssetKind = AssetKind::MeshletMesh;
    pub const SCHEMA_VERSION: u32 = VMESH_ASSET_SCHEMA_VERSION;

    /// Validate + adopt already-serialized bytes.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        parse(&bytes)?;
        Ok(Self { bytes })
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// A random-access view over the image.
    pub fn reader(&self) -> VgeomAssetView<'_> {
        VgeomAssetReader::new(self.bytes.as_slice()).expect("image validated at construction")
    }
}

impl std::fmt::Debug for VgeomAssetImage {
    /// Summarizes; never dumps the (possibly hundred-MB) payload.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let r = self.reader();
        f.debug_struct("VgeomAssetImage")
            .field("bytes", &self.bytes.len())
            .field("pages", &r.pages().len())
            .field("meshlets", &r.header().meshlet_count)
            .finish()
    }
}

// ── reader ──────────────────────────────────────────────────────────────────

/// Random access over a v2 `.inf_vmesh` payload.
///
/// Generic over the byte source so one reader serves every backing: an owned
/// `Vec<u8>` (a loose file), a `&[u8]` borrowed from
/// [`PackReader::read_ref`](inf_asset::PackReader::read_ref)'s mapping, or a `Cow`
/// straight off it. The header + directory are parsed once at construction; every
/// [`page_sections`](Self::page_sections) after that is five slices.
#[derive(Debug, Clone)]
pub struct VgeomAssetReader<B> {
    bytes: B,
    header: VgeomAssetHeader,
    pages: Vec<VgeomPageEntry>,
}

/// A [`VgeomAssetReader`] borrowing its bytes (the pack-mapping case).
pub type VgeomAssetView<'a> = VgeomAssetReader<&'a [u8]>;

impl<B: AsRef<[u8]>> VgeomAssetReader<B> {
    /// Parse + validate a payload image.
    pub fn new(bytes: B) -> Result<Self> {
        let (header, pages) = parse(bytes.as_ref())?;
        Ok(Self {
            bytes,
            header,
            pages,
        })
    }

    #[inline]
    pub fn bytes(&self) -> &[u8] {
        self.bytes.as_ref()
    }

    #[inline]
    pub fn header(&self) -> &VgeomAssetHeader {
        &self.header
    }

    /// The page directory, **coarsest first** (page 0 = the roots).
    #[inline]
    pub fn pages(&self) -> &[VgeomPageEntry] {
        &self.pages
    }

    /// The borrowed sections of page `index`.
    pub fn page_sections(&self, index: usize) -> Option<VgeomPageSections<'_>> {
        let e = self.pages.get(index)?;
        let b = self.bytes.as_ref();
        // Bounds were validated in `parse`.
        let take = |off: u64, len: u64| &b[off as usize..(off + len) as usize];
        Some(VgeomPageSections {
            indices: take(e.indices_off, e.meshlet_count as u64 * 4),
            meshlets: take(
                e.meshlets_off,
                e.meshlet_count as u64 * MESHLET_REC_LEN as u64,
            ),
            mlverts: take(e.mlverts_off, e.mlvert_count as u64 * 4),
            mltris: take(e.mltris_off, e.mltri_count as u64),
            vertices: take(
                e.vertices_off,
                e.vertex_count as u64 * vertex_rec_len(self.header.schema_version) as u64,
            ),
            tiles: take(e.tiles_off, e.tile_count as u64 * TILE_REF_LEN as u64),
        })
    }

    /// The DAG groups (decoded from the trailing section).
    pub fn groups(&self) -> Result<Vec<Group>> {
        let b = self.bytes.as_ref();
        let s = &b[self.header.groups_off as usize
            ..(self.header.groups_off + self.header.groups_len) as usize];
        if s.is_empty() {
            return Ok(Vec::new());
        }
        bincode::serde::decode_from_slice::<Vec<Group>, _>(s, inf_asset::bincode_config())
            .map(|(v, _)| v)
            .map_err(|e| VgeomAssetError::GroupsDecode(e.to_string()))
    }

    /// Materialize the whole [`VgeomMesh`] — the **non-streaming** door, for the
    /// classic discrete-LOD fallback path, tooling and tests.
    ///
    /// Inverts [`build_vgeom_asset`] up to the container's internal vertex
    /// permutation: meshlets, micro-triangle bytes, level ranges, groups and
    /// bounds come back identical, and `meshlet_vertices` comes back renumbered
    /// into the image's vertex order (the same geometry).
    pub fn to_mesh(&self) -> Result<VgeomMesh> {
        let h = &self.header;
        let n = h.meshlet_count as usize;
        let mut vertices: Vec<VgeomVertex> = Vec::with_capacity(h.vertex_count as usize);
        // Meshlets are scattered across pages; rebuild in global index order so the
        // micro-index concatenation matches `build_vgeom`'s own layout.
        let mut recs: Vec<Option<(MeshletRec, usize, u32)>> = vec![None; n];
        for (p, _) in self.pages.iter().enumerate() {
            let s = self
                .page_sections(p)
                .ok_or(VgeomAssetError::SectionOutOfBounds {
                    page: p,
                    section: "page",
                })?;
            read_vertices(h.schema_version, s.vertices, &mut vertices);
            let idx = bytemuck::cast_slice::<u8, u32>(s.indices);
            for (k, rec) in bytemuck::cast_slice::<u8, MeshletRec>(s.meshlets)
                .iter()
                .enumerate()
            {
                let g = *idx.get(k).ok_or(VgeomAssetError::Malformed(
                    "page index list is shorter than its meshlet list".into(),
                ))? as usize;
                if g >= n {
                    return Err(VgeomAssetError::Malformed(format!(
                        "page {p} names meshlet {g} of {n}"
                    )));
                }
                recs[g] = Some((*rec, p, k as u32));
            }
        }
        vertices.resize(h.vertex_count as usize, VgeomVertex::default());

        let mut meshlets: Vec<Meshlet> = Vec::with_capacity(n);
        let mut meshlet_vertices: Vec<u32> = Vec::new();
        let mut meshlet_triangles: Vec<u8> = Vec::new();
        let mut level_of: Vec<(u8, u32)> = Vec::with_capacity(n);
        for (g, slot) in recs.iter().enumerate() {
            let (rec, p, _) = slot.ok_or(VgeomAssetError::Malformed(format!(
                "meshlet {g} is missing from every page"
            )))?;
            let s = self.page_sections(p).expect("page validated above");
            let vertex_offset = meshlet_vertices.len() as u32;
            let lo = rec.vertex_offset as usize;
            meshlet_vertices.extend_from_slice(
                &bytemuck::cast_slice::<u8, u32>(s.mlverts)[lo..lo + rec.vertex_count as usize],
            );
            let triangle_offset = meshlet_triangles.len() as u32;
            let lo = rec.triangle_offset as usize;
            meshlet_triangles
                .extend_from_slice(&s.mltris[lo..lo + rec.triangle_count as usize * 3]);
            meshlets.push(Meshlet {
                vertex_offset,
                vertex_count: rec.vertex_count,
                triangle_offset,
                triangle_count: rec.triangle_count,
                center: rec.center,
                radius: rec.radius,
                cone_axis: rec.cone_axis,
                cone_cutoff: rec.cone_cutoff,
                group: rec.group,
                lod_level: rec.lod_level as u8,
                error: rec.error,
                parent_error: rec.parent_error,
            });
            level_of.push((rec.lod_level as u8, g as u32));
        }

        // `LevelRange`s: the global meshlet order is level-major coarsest-first
        // (`build_vgeom::assemble`), so each level is one contiguous run.
        let mut levels: Vec<LevelRange> = Vec::new();
        for (lod, g) in level_of {
            match levels.last_mut() {
                Some(last) if last.lod_level == lod => last.meshlet_count += 1,
                _ => levels.push(LevelRange {
                    lod_level: lod,
                    meshlet_start: g,
                    meshlet_count: 1,
                }),
            }
        }

        Ok(VgeomMesh {
            schema_version: VgeomMesh::CURRENT_VERSION,
            vertices,
            meshlets,
            meshlet_vertices,
            meshlet_triangles,
            groups: self.groups()?,
            levels,
            center: h.center,
            radius: h.radius,
        })
    }
}

/// Append one page's stored vertex records, widening a **v2** record to the v3
/// shape as it goes.
///
/// The v2 record is position + normal + uv and carries no tangent, which is not
/// an approximation of one — it is the honest statement that the container it
/// came from had none, so the widened record takes [`model::NO_TANGENT`] and
/// every consumer falls back to the cotangent frame it used before P28.2.
fn read_vertices(schema: u32, bytes: &[u8], out: &mut Vec<VgeomVertex>) {
    if schema >= 3 {
        out.extend_from_slice(bytemuck::cast_slice::<u8, VgeomVertex>(bytes));
        return;
    }
    for r in bytes.chunks_exact(VERTEX_REC_LEN_V2) {
        let f = |i: usize| f32::from_le_bytes(r[i * 4..i * 4 + 4].try_into().unwrap());
        out.push(VgeomVertex {
            position: [f(0), f(1), f(2)],
            normal: [f(3), f(4), f(5)],
            uv: [f(6), f(7)],
            tangent: crate::model::NO_TANGENT,
        });
    }
}

/// Parse + validate the header and page directory of a payload image.
fn parse(data: &[u8]) -> Result<(VgeomAssetHeader, Vec<VgeomPageEntry>)> {
    if (data.len() as u64) < HEADER_LEN {
        return Err(VgeomAssetError::TooShort);
    }
    if data[0..8] != VMESH_ASSET_MAGIC {
        return Err(VgeomAssetError::BadMagic);
    }
    let u32_at = |o: usize| u32::from_le_bytes(data[o..o + 4].try_into().unwrap());
    let f32_at = |o: usize| f32::from_le_bytes(data[o..o + 4].try_into().unwrap());
    let u64_at = |o: usize| u64::from_le_bytes(data[o..o + 8].try_into().unwrap());

    let schema_version = u32_at(8);
    if schema_version > VMESH_ASSET_SCHEMA_VERSION {
        return Err(VgeomAssetError::SchemaTooNew {
            found: schema_version,
            current: VMESH_ASSET_SCHEMA_VERSION,
        });
    }
    let page_count = u32_at(12);
    let header = VgeomAssetHeader {
        schema_version,
        page_count,
        meshlet_count: u32_at(16),
        vertex_count: u32_at(20),
        max_tri: u32_at(24),
        max_lod: u32_at(28),
        center: [f32_at(36), f32_at(40), f32_at(44)],
        radius: f32_at(48),
        page_base: u64_at(56),
        groups_off: u64_at(64),
        groups_len: u64_at(72),
        total_len: u64_at(80),
    };

    // Every record the header claims has to be *stored*, so the payload length
    // bounds the counts from above. Without this a doctored header can name
    // `u32::MAX` vertices and `to_mesh`'s `Vec::with_capacity` asks the allocator
    // for ~127 GiB — an abort, not an error, from a file the process merely opened.
    // (The directory walk below bounds `meshlet_count` too, since every meshlet
    // lives in some page's bounds-checked section; vertices need this because a
    // page may legitimately store fewer than the header's count when some vertex
    // is referenced by nothing.)
    let payload_len = data.len() as u64;
    let vrec = vertex_rec_len(schema_version) as u64;
    for (count, stride, what) in [
        (header.vertex_count, vrec, "vertices"),
        (header.meshlet_count, MESHLET_REC_LEN as u64, "meshlets"),
    ] {
        if count as u64 * stride > payload_len {
            return Err(VgeomAssetError::Malformed(format!(
                "header claims {count} {what} ({} B) in a {payload_len} B payload",
                count as u64 * stride
            )));
        }
    }

    let dir_end = HEADER_LEN + PAGE_ENTRY_LEN * page_count as u64;
    if payload_len < dir_end {
        return Err(VgeomAssetError::TooShort);
    }
    if page_count > 0
        && (header.page_base < dir_end || !header.page_base.is_multiple_of(SECTION_ALIGN))
    {
        return Err(VgeomAssetError::Malformed(format!(
            "page_base {} is before the directory end {dir_end} or misaligned",
            header.page_base
        )));
    }
    let end = data.len() as u64;
    if header
        .groups_off
        .checked_add(header.groups_len)
        .is_none_or(|e| e > end)
    {
        return Err(VgeomAssetError::Malformed(
            "groups section is out of bounds".into(),
        ));
    }

    let mut pages = Vec::with_capacity(page_count as usize);
    let mut next_vertex = 0u32;
    let mut meshlets_seen: u64 = 0;
    for i in 0..page_count as usize {
        let b = HEADER_LEN as usize + i * PAGE_ENTRY_LEN as usize;
        let e = VgeomPageEntry {
            page: u32_at(b),
            meshlet_count: u32_at(b + 4),
            vertex_start: u32_at(b + 8),
            vertex_count: u32_at(b + 12),
            mlvert_count: u32_at(b + 16),
            mltri_count: u32_at(b + 20),
            floor_lod: u32_at(b + 24),
            lod: u32_at(b + 28),
            max_parent_error: f32_at(b + 32),
            min_error: f32_at(b + 36),
            // v3 fields. A v2 directory wrote zeros in both lanes, so a v2 image
            // parses as "no pairing" — which is the truth about it, not a
            // default standing in for one.
            tile_count: u32_at(b + 40),
            indices_off: u64_at(b + 48),
            meshlets_off: u64_at(b + 56),
            mlverts_off: u64_at(b + 64),
            mltris_off: u64_at(b + 72),
            vertices_off: u64_at(b + 80),
            tiles_off: u64_at(b + 88),
        };
        if schema_version < 3 && (e.tile_count != 0 || e.tiles_off != 0) {
            return Err(VgeomAssetError::Malformed(format!(
                "page {i} of a v{schema_version} image carries a v3 tiles section"
            )));
        }
        if e.page != i as u32 {
            return Err(VgeomAssetError::Malformed(format!(
                "page directory entry {i} is labelled {}",
                e.page
            )));
        }
        // Vertex blocks must tile a prefix with no gap and no overlap — the
        // property prefix residency rests on.
        if e.vertex_start != next_vertex {
            return Err(VgeomAssetError::Malformed(format!(
                "page {i} vertex block starts at {}, expected {next_vertex}",
                e.vertex_start
            )));
        }
        next_vertex = e
            .vertex_start
            .checked_add(e.vertex_count)
            .ok_or_else(|| VgeomAssetError::Malformed("vertex block overflows".into()))?;
        meshlets_seen += e.meshlet_count as u64;
        for (off, len, section) in [
            (e.indices_off, e.meshlet_count as u64 * 4, "indices"),
            (
                e.meshlets_off,
                e.meshlet_count as u64 * MESHLET_REC_LEN as u64,
                "meshlets",
            ),
            (e.mlverts_off, e.mlvert_count as u64 * 4, "mlverts"),
            (e.mltris_off, e.mltri_count as u64, "mltris"),
            (e.vertices_off, e.vertex_count as u64 * vrec, "vertices"),
            // A v2 image has no tiles lane at all — the guard above proved both
            // its fields are zero, and a zero offset is not a section to bound.
            // A v3 page always carries a real, aligned offset here, empty or not,
            // so `(0, 0)` for v2 is the one pair this loop must not check.
            if schema_version >= 3 {
                (
                    e.tiles_off,
                    e.tile_count as u64 * TILE_REF_LEN as u64,
                    "tiles",
                )
            } else {
                (header.page_base, 0, "tiles")
            },
        ] {
            if !off.is_multiple_of(SECTION_ALIGN)
                || off < header.page_base
                || off.checked_add(len).is_none_or(|x| x > end)
            {
                return Err(VgeomAssetError::SectionOutOfBounds { page: i, section });
            }
        }
        pages.push(e);
    }
    if next_vertex > header.vertex_count || meshlets_seen != header.meshlet_count as u64 {
        return Err(VgeomAssetError::Malformed(
            "page directory does not account for exactly the header's meshlets/vertices".into(),
        ));
    }
    validate_records(data, &pages)?;
    Ok((header, pages))
}

/// Check every stored meshlet record's micro-index ranges against its page's
/// sections — the **read-side counterpart** of the build-side [`validate`].
///
/// The build side proves the ranges are sound for a mesh *this process* is about
/// to write. That says nothing about a payload that arrived from disk, and
/// [`VgeomAssetReader::to_mesh`] slices with those offsets directly: a doctored
/// `vertex_offset` panics on a 64-bit host and, on `wasm32` where `as usize`
/// truncates a `u32` no further, still yields the wrong slice. Both are reachable
/// from a shipped pack through `classic_vgeom`'s `to_mesh().ok()?`, which cannot
/// catch a panic. So the bounds are established **once, at parse**, and every
/// slice after that is unchecked-safe — the same discipline `inf_asset::pack`
/// applies to blob offsets.
///
/// `O(meshlets)`: a linear scan of records already in the page cache, microseconds
/// even for a six-figure meshlet count, and paid once when the asset is indexed
/// rather than on every fetch.
fn validate_records(data: &[u8], pages: &[VgeomPageEntry]) -> Result<()> {
    for (page, e) in pages.iter().enumerate() {
        let lo = e.meshlets_off as usize;
        let hi = lo + e.meshlet_count as usize * MESHLET_REC_LEN;
        // The section itself was bounds-checked above; this cast is exact because
        // `MeshletRec` is `Pod` and the slice length is a multiple of its size.
        let recs: &[MeshletRec] = bytemuck::cast_slice(&data[lo..hi]);
        for (index, r) in recs.iter().enumerate() {
            let bad = |what| VgeomAssetError::RecordOutOfBounds { page, index, what };
            let v_end = u64::from(r.vertex_offset)
                .checked_add(u64::from(r.vertex_count))
                .ok_or_else(|| bad("vertex"))?;
            if v_end > u64::from(e.mlvert_count) {
                return Err(bad("vertex"));
            }
            let t_end = u64::from(r.triangle_offset)
                .checked_add(u64::from(r.triangle_count).saturating_mul(3))
                .ok_or_else(|| bad("triangle"))?;
            if t_end > u64::from(e.mltri_count) {
                return Err(bad("triangle"));
            }
        }
    }
    Ok(())
}

// ── the runtime source (pack mapping / owned bytes / in-memory mesh) ─────────

/// Where a [`VgeomSource`] gets its bytes.
enum Backing {
    /// Zero-copy: slice straight out of the pack mapping on every fetch — the
    /// `PackTileStore` precedent, and the reason [`AssetKind::MeshletMesh`] cooks
    /// uncompressed.
    Pack(Arc<PackReader>, AssetId),
    /// A loose file, an in-memory build, a lifted v1 payload, or the
    /// compressed-entry fallback.
    ///
    /// [`AlignedBytes`], not `Vec<u8>`: a section's 16-byte alignment inside the
    /// payload only becomes an aligned *address* if the base is aligned too, and
    /// `bytemuck::cast_slice::<_, MeshletRec>` on a misaligned base panics. A bare
    /// `Vec` is 1-byte aligned and would happen to work most of the time on a
    /// 64-bit host — the P16.1 reasoning, and the reason `inf_asset` carries the
    /// alignment in the buffer type rather than in a comment.
    Owned(inf_asset::AlignedBytes),
}

/// A lazily-indexed `.inf_vmesh` the renderer pages meshlets out of.
///
/// Construction parses **only** the header + page directory — a few hundred
/// bytes — so a scene with a thousand vmeshes costs a thousand directory parses,
/// not a thousand full decodes. Every subsequent
/// [`with_page_sections`](Self::with_page_sections) is five sub-slices of the
/// mapping.
pub struct VgeomSource {
    backing: Backing,
    header: VgeomAssetHeader,
    pages: Vec<VgeomPageEntry>,
}

impl VgeomSource {
    /// Index the `.inf_vmesh` asset `guid` inside a cooked pack, **without
    /// decoding it**.
    ///
    /// A pack that stores the entry compressed (an older cook, or a hand-built
    /// pack) still works: the payload is decompressed **once** here into an owned
    /// buffer and served from there — slower to open, identical afterwards.
    pub fn open_pack(reader: Arc<PackReader>, guid: AssetId) -> std::result::Result<Self, String> {
        let payload = reader
            .read_ref(guid)
            .map_err(|e| format!("read vmesh {guid}: {e}"))?;
        let borrowed = matches!(payload, Cow::Borrowed(_));
        if !is_current(payload.as_ref()) {
            // A v1 (bincode) or v2 (narrow vertex record, no pairing) entry has to
            // be lifted into a v3 image, which means owning bytes either way.
            let mesh = lift_to_mesh(payload.as_ref()).map_err(|e| format!("vmesh {guid}: {e}"))?;
            return Self::from_mesh(&mesh).map_err(|e| format!("vmesh {guid}: {e}"));
        }
        let (header, pages) = {
            let r = VgeomAssetReader::new(payload.as_ref())
                .map_err(|e| format!("vmesh {guid}: {e}"))?;
            (*r.header(), r.pages().to_vec())
        };
        let backing = if borrowed {
            drop(payload);
            Backing::Pack(reader, guid)
        } else {
            Backing::Owned(inf_asset::AlignedBytes::copy_from(payload.as_ref()))
        };
        Ok(Self {
            backing,
            header,
            pages,
        })
    }

    /// Index a payload read from a loose file (the editor's cold side), lifting a
    /// **v1** bincode or **v2** paged payload if that is what it is.
    ///
    /// The lift is the P18.2 arrangement, one version on: an older payload is
    /// materialized and re-laid-out as a current image, so the streaming code has
    /// exactly one shape to be right about. Slower to open, identical afterwards
    /// — and a lifted asset has no tangents and no tile pairing, because the
    /// container it came from stored neither.
    pub fn from_payload(bytes: Vec<u8>) -> std::result::Result<Self, String> {
        if !is_current(&bytes) {
            let mesh = lift_to_mesh(&bytes)?;
            return Self::from_mesh(&mesh);
        }
        Self::from_image(bytes)
    }

    /// Index an already-v2 image.
    pub fn from_image(bytes: Vec<u8>) -> std::result::Result<Self, String> {
        let bytes = inf_asset::AlignedBytes::copy_from(&bytes);
        let (header, pages) = {
            let r = VgeomAssetReader::new(bytes.as_slice()).map_err(|e| e.to_string())?;
            (*r.header(), r.pages().to_vec())
        };
        Ok(Self {
            backing: Backing::Owned(bytes),
            header,
            pages,
        })
    }

    /// Lay an in-memory [`VgeomMesh`] out as a v3 image **with no tile pairing**
    /// and index it.
    ///
    /// This is what makes "one reader for everything" true: an editor build, a
    /// test fixture and a lifted v1 payload all reach the renderer through the
    /// **same** paged path a cooked pack does, so the streaming code has no second
    /// shape to be right about. A pairing comes from the cook, which is the only
    /// place a mesh's materials are visible — see [`Self::from_mesh_paired`].
    pub fn from_mesh(mesh: &VgeomMesh) -> std::result::Result<Self, String> {
        Self::from_mesh_paired(mesh, &ClusterTextureSet::none())
    }

    /// [`Self::from_mesh`] with a cluster→tile pairing.
    pub fn from_mesh_paired(
        mesh: &VgeomMesh,
        textures: &ClusterTextureSet,
    ) -> std::result::Result<Self, String> {
        let image = build_vgeom_asset(mesh, textures).map_err(|e| e.to_string())?;
        Self::from_image(image.into_bytes())
    }

    #[inline]
    pub fn header(&self) -> &VgeomAssetHeader {
        &self.header
    }

    /// The page directory, **coarsest first** (page 0 = the roots).
    #[inline]
    pub fn pages(&self) -> &[VgeomPageEntry] {
        &self.pages
    }

    /// Meshlets across every page (the cull dispatch's per-instance width).
    #[inline]
    pub fn meshlet_count(&self) -> u32 {
        self.header.meshlet_count
    }

    /// The vertex-pulled indirect draw's `vertex_count / 3` — from the header, so
    /// the draw shape never depends on what is resident.
    #[inline]
    pub fn max_tri(&self) -> u32 {
        self.header.max_tri
    }

    /// Whole-mesh bounding sphere (local space).
    #[inline]
    pub fn bounds(&self) -> ([f32; 3], f32) {
        (self.header.center, self.header.radius)
    }

    /// Bytes if every page were resident — the asset's full VRAM cost.
    pub fn total_resident_bytes(&self) -> u64 {
        self.pages.iter().map(|p| p.resident_bytes()).sum()
    }

    /// The payload bytes, borrowed from whichever backing holds them.
    pub fn payload(&self) -> Option<Cow<'_, [u8]>> {
        match &self.backing {
            Backing::Pack(reader, guid) => reader.read_ref(*guid).ok(),
            Backing::Owned(v) => Some(Cow::Borrowed(v.as_slice())),
        }
    }

    /// The sections of page `index`, applied to `f` while the borrow lives.
    ///
    /// A closure rather than a returned slice because the pack backing's borrow is
    /// tied to the `read_ref` guard, which cannot outlive this call.
    pub fn with_page_sections<R>(
        &self,
        index: usize,
        f: impl FnOnce(VgeomPageSections<'_>) -> R,
    ) -> Option<R> {
        let payload = self.payload()?;
        let r = VgeomAssetReader::new(payload.as_ref()).ok()?;
        let s = r.page_sections(index)?;
        Some(f(s))
    }

    /// Materialize the whole mesh — the classic discrete-LOD fallback's door.
    pub fn to_mesh(&self) -> std::result::Result<VgeomMesh, String> {
        let payload = self
            .payload()
            .ok_or_else(|| "vmesh payload unavailable".to_string())?;
        VgeomAssetReader::new(payload.as_ref())
            .and_then(|r| r.to_mesh())
            .map_err(|e| e.to_string())
    }
}

impl std::fmt::Debug for VgeomSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VgeomSource")
            .field(
                "backing",
                &match &self.backing {
                    Backing::Pack(_, guid) => format!("pack:{guid}"),
                    Backing::Owned(v) => format!("owned:{} bytes", v.len()),
                },
            )
            .field("pages", &self.pages.len())
            .field("meshlets", &self.header.meshlet_count)
            .finish()
    }
}

/// Whether `bytes` look like a **paged** image of any version (as opposed to a v1
/// bincode `VgeomMesh`, whose first four bytes are `schema_version = 1`).
pub fn is_v2(bytes: &[u8]) -> bool {
    bytes.len() >= 8 && bytes[0..8] == VMESH_ASSET_MAGIC
}

/// The container version of a paged image, or `None` for a v1 bincode payload.
pub fn container_version(bytes: &[u8]) -> Option<u32> {
    if !is_v2(bytes) || bytes.len() < 12 {
        return None;
    }
    Some(u32::from_le_bytes(bytes[8..12].try_into().unwrap()))
}

/// Whether `bytes` are a paged image at the **current** container version, i.e.
/// one the streaming path can read without a lift.
pub fn is_current(bytes: &[u8]) -> bool {
    container_version(bytes) == Some(VMESH_ASSET_SCHEMA_VERSION)
}

/// Materialize an **older** payload — a v1 bincode [`VgeomMesh`] or a v2 paged
/// image — so it can be re-laid-out as a current image.
fn lift_to_mesh(bytes: &[u8]) -> std::result::Result<VgeomMesh, String> {
    match container_version(bytes) {
        None => decode_bincode(bytes),
        Some(v) if v < VMESH_ASSET_SCHEMA_VERSION => VgeomAssetReader::new(bytes)
            .and_then(|r| r.to_mesh())
            .map_err(|e| format!("lift v{v} vmesh: {e}")),
        Some(v) => Err(format!(
            "vmesh container v{v} is not older than v{VMESH_ASSET_SCHEMA_VERSION}"
        )),
    }
}

/// Decode a **bincode** payload: the bare [`VgeomMesh`] every pack cooked before
/// P18.2 carries.
///
/// `VgeomMesh::schema_version` **1**'s vertices are position + normal + uv, which
/// is a *different* bincode shape from schema 2's, which carries the tangent
/// word. bincode is positional, so a v1 blob decoded into today's struct would
/// desync at the first vertex and produce garbage without erroring. So schema 1
/// decodes through a **frozen shadow record** ([`v1_records`]) and is converted —
/// the same discipline P17's ladder-local frozen literals apply, and the only
/// way a committed fixture stays readable across a vertex-format change.
fn decode_bincode(bytes: &[u8]) -> std::result::Result<VgeomMesh, String> {
    let cfg = inf_asset::bincode_config();
    match inf_asset::peek_schema_version(bytes, Some(VgeomMesh::CURRENT_VERSION)) {
        Some(1) => bincode::serde::decode_from_slice::<v1_records::VgeomMeshV1, _>(bytes, cfg)
            .map(|(m, _)| m.into_current())
            .map_err(|e| format!("decode v1 vmesh: {e}")),
        _ => inf_asset::decode::<VgeomMesh>(bytes).map_err(|e| format!("decode vmesh: {e}")),
    }
}

/// The **frozen** v1 bincode records. Never edit these to follow a change to the
/// live model: their whole job is to keep describing bytes that already exist.
mod v1_records {
    use super::{Group, LevelRange, Meshlet, VgeomMesh, VgeomVertex};
    use serde::Deserialize;

    #[derive(Deserialize)]
    pub struct VgeomVertexV1 {
        pub position: [f32; 3],
        pub normal: [f32; 3],
        pub uv: [f32; 2],
    }

    #[derive(Deserialize)]
    pub struct VgeomMeshV1 {
        /// Decoded and discarded: the caller already read it through
        /// `peek_schema_version` to choose this record, and bincode is
        /// positional, so the field has to be *decoded* whether or not it is
        /// read.
        #[allow(dead_code)]
        pub schema_version: u32,
        pub vertices: Vec<VgeomVertexV1>,
        pub meshlets: Vec<Meshlet>,
        pub meshlet_vertices: Vec<u32>,
        pub meshlet_triangles: Vec<u8>,
        pub groups: Vec<Group>,
        pub levels: Vec<LevelRange>,
        pub center: [f32; 3],
        pub radius: f32,
    }

    impl VgeomMeshV1 {
        pub fn into_current(self) -> VgeomMesh {
            VgeomMesh {
                schema_version: VgeomMesh::CURRENT_VERSION,
                vertices: self
                    .vertices
                    .into_iter()
                    .map(|v| VgeomVertex {
                        position: v.position,
                        normal: v.normal,
                        uv: v.uv,
                        tangent: crate::model::NO_TANGENT,
                    })
                    .collect(),
                meshlets: self.meshlets,
                meshlet_vertices: self.meshlet_vertices,
                meshlet_triangles: self.meshlet_triangles,
                groups: self.groups,
                levels: self.levels,
                center: self.center,
                radius: self.radius,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::dense_mesh;

    /// The container is a re-layout, not a re-mesh: everything except the internal
    /// vertex permutation comes back identical, and the geometry is the same
    /// triangles at the same positions.
    #[test]
    fn round_trips_the_mesh() {
        let m = dense_mesh(24);
        let image = build_vgeom_asset(&m, &ClusterTextureSet::none()).expect("build");
        let back = image.reader().to_mesh().expect("materialize");
        assert_eq!(back.meshlets, m.meshlets, "meshlets survive the re-layout");
        assert_eq!(back.meshlet_triangles, m.meshlet_triangles);
        assert_eq!(back.levels, m.levels);
        assert_eq!(back.groups, m.groups);
        assert_eq!(back.center, m.center);
        assert_eq!(back.radius, m.radius);
        assert_eq!(back.vertices.len(), m.vertices.len());
        // Same geometry under the permutation.
        for i in 0..m.meshlets.len() {
            for t in 0..m.meshlets[i].triangle_count as usize {
                let a = m.triangle(i, t).map(|v| m.vertices[v as usize]);
                let b = back.triangle(i, t).map(|v| back.vertices[v as usize]);
                assert_eq!(a, b, "meshlet {i} triangle {t}");
            }
        }
    }

    #[test]
    fn build_is_byte_deterministic() {
        let m = dense_mesh(20);
        let a = build_vgeom_asset(&m, &ClusterTextureSet::none()).unwrap().into_bytes();
        let b = build_vgeom_asset(&m, &ClusterTextureSet::none()).unwrap().into_bytes();
        assert_eq!(a, b, "two builds of one mesh are byte-identical");
    }

    #[test]
    fn sections_are_aligned_and_in_bounds() {
        let m = dense_mesh(24);
        let image = build_vgeom_asset(&m, &ClusterTextureSet::none()).unwrap();
        let r = image.reader();
        assert!(r.pages().len() >= 3, "expected a multi-page DAG");
        for (i, e) in r.pages().iter().enumerate() {
            for off in [
                e.indices_off,
                e.meshlets_off,
                e.mlverts_off,
                e.mltris_off,
                e.vertices_off,
            ] {
                assert_eq!(off % SECTION_ALIGN, 0, "page {i} section misaligned");
            }
            let s = r.page_sections(i).expect("sections");
            assert_eq!(s.meshlets.len(), e.meshlet_count as usize * MESHLET_REC_LEN);
            assert_eq!(s.indices.len(), e.meshlet_count as usize * 4);
            assert_eq!(s.vertices.len(), e.vertex_count as usize * VERTEX_REC_LEN);
            assert_eq!(s.mlverts.len(), e.mlvert_count as usize * 4);
            assert_eq!(s.mltris.len(), e.mltri_count as usize);
        }
    }

    /// Page 0 holds **every** root, and no other page holds one. This is the
    /// never-a-hole guarantee at the format level: page 0 is never evicted, so
    /// every root-to-leaf path always has a resident meshlet.
    #[test]
    fn page_zero_is_exactly_the_roots() {
        let m = dense_mesh(24);
        let image = build_vgeom_asset(&m, &ClusterTextureSet::none()).unwrap();
        let r = image.reader();
        let s0 = r.page_sections(0).unwrap();
        let page0: std::collections::BTreeSet<u32> = bytemuck::cast_slice::<u8, u32>(s0.indices)
            .iter()
            .copied()
            .collect();
        let roots: std::collections::BTreeSet<u32> = m
            .meshlets
            .iter()
            .enumerate()
            .filter(|(_, x)| x.is_root())
            .map(|(i, _)| i as u32)
            .collect();
        assert_eq!(page0, roots, "page 0 == the root set");
        assert!(!roots.is_empty());
        assert!(r.pages()[0].max_parent_error.is_infinite());
        for i in 1..r.pages().len() {
            let s = r.page_sections(i).unwrap();
            for &g in bytemuck::cast_slice::<u8, u32>(s.indices) {
                assert!(!m.meshlets[g as usize].is_root(), "page {i} holds a root");
            }
            assert!(r.pages()[i].max_parent_error.is_finite());
        }
    }

    /// The per-page vertex blocks tile `[0, prefix)` with no gap and no overlap,
    /// and page `p` references only vertices inside its own prefix — the property
    /// prefix residency rests on.
    #[test]
    fn vertex_blocks_tile_the_prefix() {
        let m = dense_mesh(24);
        let image = build_vgeom_asset(&m, &ClusterTextureSet::none()).unwrap();
        let r = image.reader();
        let mut prefix = 0u32;
        for (i, e) in r.pages().iter().enumerate() {
            assert_eq!(e.vertex_start, prefix, "vertex blocks must be contiguous");
            prefix = e.vertex_start + e.vertex_count;
            let s = r.page_sections(i).unwrap();
            for &v in bytemuck::cast_slice::<u8, u32>(s.mlverts) {
                assert!(
                    v < prefix,
                    "page {i} references vertex {v} outside its prefix {prefix}"
                );
            }
        }
        assert!(prefix <= r.header().vertex_count);
    }

    /// The **frozen v1 fixture** loads, forever.
    ///
    /// `v1_payload_loads_forever` below re-encodes a mesh at runtime, which pins
    /// "today's writer round-trips through today's reader" — a real property, but
    /// not the one v1 back-compat is about. This test reads bytes that were
    /// committed and are never regenerated, so it fails the day the v1 lift stops
    /// understanding a payload that is already in somebody's shipped pack.
    ///
    /// **Provenance:** produced by `regenerate_frozen_v1_fixture` (ignored by
    /// default, in this module) against the P18.1-era encoding — `inf_asset::encode`
    /// of `build_vgeom` over the 12x12 displaced grid `test_support::dense_mesh(12)`
    /// built *at that time*, with `BuildParams::default()`. **The generator has
    /// since changed** — `dense_mesh` was ported off `std` trig onto
    /// `psin64`/`pcos64` (portable fixtures), so re-running the regenerator today
    /// would emit different bytes. That is not drift, it is the point: this
    /// fixture pins **BYTES**, not the generator, and the assertions below only
    /// ever read the committed file. Leave it exactly as it is.
    ///
    /// To regenerate it you must first establish that the *old* bytes are
    /// genuinely unloadable rather than merely inconvenient, and then say so in
    /// this comment; it must never be re-blessed from the current writer, which
    /// emits the v2 paged image and would turn this gate into a tautology.
    #[test]
    fn loads_the_frozen_v1_fixture() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/v1_dense12.inf_vmesh");
        let bytes = std::fs::read(&path).expect("committed v1 fixture present");
        assert!(!is_v2(&bytes), "the fixture must be a genuine v1 payload");
        // A bincode `VgeomMesh` opens with its `schema_version`, varint-encoded —
        // one byte for 1. That is also why the v1 magic sniff can never collide.
        assert_eq!(
            bytes[0], 1,
            "the fixture opens with schema_version = 1, varint-encoded"
        );

        let src = VgeomSource::from_payload(bytes.clone()).expect("the v1 lift still works");
        // It is a real DAG, not an empty shell that would pass vacuously.
        assert!(src.meshlet_count() >= 2, "fixture carries meshlets");
        assert!(
            src.pages().len() >= 2,
            "fixture carries pages beyond the roots"
        );
        assert!(src.pages()[0].is_root_page());
        assert!(src.max_tri() > 0);
        assert!(src.total_resident_bytes() > 0);

        // And it materializes back to exactly the DAG the old bytes describe.
        // Through the crate's own legacy door, because since P28.2 `VgeomVertex`
        // carries a tangent word and bincode is positional: `inf_asset::decode`
        // into today's struct would desync at the first vertex. The FROZEN shadow
        // record is what reads these bytes, and its being exercised here — on the
        // committed file rather than on something this test wrote — is the point.
        let direct: VgeomMesh = decode_bincode(&bytes).expect("v1 decodes through the shadow");
        assert!(
            direct
                .vertices
                .iter()
                .all(|v| v.tangent == crate::model::NO_TANGENT),
            "a v1 payload has no tangents, and the lift must not invent any"
        );
        let lifted = src.to_mesh().expect("materialize the lifted image");
        assert_eq!(lifted.meshlets, direct.meshlets);
        assert_eq!(lifted.meshlet_triangles, direct.meshlet_triangles);
        assert_eq!(lifted.levels, direct.levels);
        assert_eq!(lifted.groups, direct.groups);
        assert_eq!(lifted.center, direct.center);
        assert_eq!(lifted.radius, direct.radius);
        assert_eq!(lifted.vertices.len(), direct.vertices.len());
        // Same geometry under the container's internal vertex permutation.
        for i in 0..direct.meshlets.len() {
            for t in 0..direct.meshlets[i].triangle_count as usize {
                let a = direct.triangle(i, t).map(|v| direct.vertices[v as usize]);
                let b = lifted.triangle(i, t).map(|v| lifted.vertices[v as usize]);
                assert_eq!(a, b, "meshlet {i} triangle {t}");
            }
        }
    }

    /// A **v1** payload — a bare bincode `VgeomMesh`, which is what every pack
    /// cooked before P18.2 carries — still opens, forever.
    #[test]
    fn v1_payload_loads_forever() {
        let m = dense_mesh(20);
        let v1 = inf_asset::encode(&m).expect("v1 encode");
        assert!(!is_v2(&v1), "a v1 payload must not look like v2");
        let src = VgeomSource::from_payload(v1).expect("v1 lifts");
        assert_eq!(src.meshlet_count(), m.meshlets.len() as u32);
        let back = src.to_mesh().expect("materialize");
        assert_eq!(back.meshlets, m.meshlets);
        assert_eq!(back.levels, m.levels);
    }

    #[test]
    fn rejects_bad_magic_short_and_newer_schema() {
        assert_eq!(
            VgeomAssetImage::from_bytes(vec![0u8; 256]).unwrap_err(),
            VgeomAssetError::BadMagic
        );
        assert_eq!(
            VgeomAssetImage::from_bytes(vec![0u8; 8]).unwrap_err(),
            VgeomAssetError::TooShort
        );
        let m = dense_mesh(12);
        let mut bytes = build_vgeom_asset(&m, &ClusterTextureSet::none()).unwrap().into_bytes();
        bytes[8..12].copy_from_slice(&99u32.to_le_bytes());
        assert!(matches!(
            VgeomAssetImage::from_bytes(bytes).unwrap_err(),
            VgeomAssetError::SchemaTooNew { found: 99, .. }
        ));
    }

    /// A corrupted directory is rejected rather than trusted — the streamer's
    /// blocked-set path depends on a bad page failing loudly at open.
    #[test]
    fn rejects_a_corrupt_directory() {
        let m = dense_mesh(12);
        let mut bytes = build_vgeom_asset(&m, &ClusterTextureSet::none()).unwrap().into_bytes();
        // Point page 0's meshlet section past the end of the payload.
        let off = HEADER_LEN as usize + 56;
        bytes[off..off + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(
            VgeomAssetImage::from_bytes(bytes).unwrap_err(),
            VgeomAssetError::SectionOutOfBounds { page: 0, .. }
        ));
    }

    /// A doctored meshlet record is rejected at **parse**, not at the slice.
    ///
    /// `to_mesh` indexes its page's `mlverts` / `mltris` with the record's own
    /// offsets. Left unchecked those are attacker-controlled indices out of a file:
    /// a panic on a 64-bit host, and a silently wrong slice on `wasm32`. Both are
    /// reachable from a shipped pack through `classic_vgeom`'s `to_mesh().ok()?`,
    /// which cannot catch a panic — so a corrupt payload has to fail at the door.
    #[test]
    fn rejects_a_record_whose_micro_index_range_escapes_its_page() {
        let m = dense_mesh(24);
        let good = build_vgeom_asset(&m, &ClusterTextureSet::none()).unwrap().into_bytes();
        // Page 1 (the coarsest non-root level) is small and definitely present.
        let page = 1usize;
        let recs_off = {
            let r = VgeomAssetReader::new(good.as_slice()).unwrap();
            assert!(r.pages().len() > page);
            r.pages()[page].meshlets_off as usize
        };
        // `MeshletRec` field order: ..., vertex_offset @32, triangle_offset @36.
        for (field_off, what) in [(32usize, "vertex"), (36, "triangle")] {
            let mut bytes = good.clone();
            let at = recs_off + field_off;
            bytes[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
            let err = VgeomAssetImage::from_bytes(bytes).unwrap_err();
            assert!(
                matches!(
                    err,
                    VgeomAssetError::RecordOutOfBounds { page: p, what: w, .. }
                        if p == page && w == what
                ),
                "expected a {what} RecordOutOfBounds, got {err}"
            );
        }
        // And an overflowing offset+count pair, not just a huge offset.
        let mut bytes = good.clone();
        let at = recs_off + 40; // vertex_count
        bytes[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            VgeomAssetImage::from_bytes(bytes).unwrap_err(),
            VgeomAssetError::RecordOutOfBounds { .. }
        ));
        // The untouched image still opens, so the test is not passing by accident.
        assert!(VgeomAssetImage::from_bytes(good).is_ok());
    }

    /// A header count is bounded by the payload that has to store it.
    ///
    /// `vertex_count` was checked only from below (the page directory must not
    /// claim *more* than it), so `u32::MAX` passed parse and `to_mesh`'s
    /// `Vec::with_capacity(u32::MAX)` then asked the allocator for ~127 GiB — an
    /// abort, from a file the process merely opened.
    #[test]
    fn rejects_header_counts_larger_than_the_payload() {
        let m = dense_mesh(16);
        let good = build_vgeom_asset(&m, &ClusterTextureSet::none()).unwrap().into_bytes();
        for off in [20usize, 16] {
            // 20 = vertex_count, 16 = meshlet_count.
            let mut bytes = good.clone();
            bytes[off..off + 4].copy_from_slice(&u32::MAX.to_le_bytes());
            let err = VgeomAssetImage::from_bytes(bytes).unwrap_err();
            assert!(
                matches!(&err, VgeomAssetError::Malformed(m) if m.contains("payload")),
                "expected a payload-length rejection at offset {off}, got {err}"
            );
        }
    }

    /// One-shot generator for the committed v1 fixture. Ignored by default: it
    /// WRITES the fixture, and re-running it against a future builder would silently
    /// re-bless the very bytes the fixture exists to freeze. See
    /// `loads_the_frozen_v1_fixture` for the provenance contract — note in
    /// particular that `dense_mesh` has been ported to portable trig *since* the
    /// committed bytes were produced, so this no longer reproduces them, and must
    /// not be run to "fix" that.
    #[test]
    #[ignore = "regenerates the frozen v1 fixture; see loads_the_frozen_v1_fixture"]
    fn regenerate_frozen_v1_fixture() {
        let m = dense_mesh(12);
        let bytes = inf_asset::encode(&m).expect("v1 encode");
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/v1_dense12.inf_vmesh");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &bytes).unwrap();
        eprintln!("wrote {} ({} bytes)", path.display(), bytes.len());
    }

    // ── v3: the cluster → tile pairing ──────────────────────────────────────

    /// A 2 048² texture with 128-texel tiles: 16×16 · 8×8 · 4×4 · 2×2 · 1×1.
    fn tex_2048(id: u8) -> ClusterTexture {
        ClusterTexture {
            id: AssetId(uuid::Uuid::from_u128(
                0xA000_0000_0000_0000_0000_0000_0000_0000 | id as u128,
            )),
            mips: vec![(16, 16), (8, 8), (4, 4), (2, 2), (1, 1)],
        }
    }

    /// **The rule, pinned**: one mip is two LOD levels, the finest geometry pairs
    /// with mip 0, and the root page pairs with the coarsest level there is.
    #[test]
    fn the_tile_mip_rule_steps_one_mip_per_two_lod_levels() {
        assert_eq!(tile_mip_for_lod(0, 5), 0, "the finest geometry gets mip 0");
        assert_eq!(tile_mip_for_lod(1, 5), 0);
        assert_eq!(tile_mip_for_lod(2, 5), 1);
        assert_eq!(tile_mip_for_lod(3, 5), 1);
        assert_eq!(tile_mip_for_lod(8, 5), 4, "clamped to the coarsest mip");
        assert_eq!(tile_mip_for_lod(u32::MAX, 5), 4, "the root page");
        // A one-level pyramid has one answer for every level.
        for lod in [0, 1, 7, u32::MAX] {
            assert_eq!(tile_mip_for_lod(lod, 1), 0);
        }
    }

    /// The pairing **covers** what the page samples: every uv of every meshlet in
    /// a page lands in a tile the page's own section names. Sampled from the
    /// vertices directly rather than from the bound the builder computed — which
    /// is the shape the residency gate's independent oracle takes at world level.
    #[test]
    fn the_cluster_pairing_covers_every_uv_a_page_touches() {
        let m = dense_mesh(24);
        let set = ClusterTextureSet {
            textures: vec![tex_2048(1), tex_2048(2)],
        };
        let img = build_vgeom_asset(&m, &set).expect("build");
        let r = img.reader();
        assert!(r.pages().len() >= 3, "a fixture with real pages");
        let mesh = r.to_mesh().expect("materialize");
        let mut total = 0usize;
        for (p, e) in r.pages().iter().enumerate() {
            let s = r.page_sections(p).expect("sections");
            let refs = s.tile_refs();
            assert_eq!(refs.len(), e.tile_count as usize);
            assert!(!refs.is_empty(), "page {p} paired nothing");
            total += refs.len();
            // Sorted, deduped, payload order — one forward scan per texture.
            for w in refs.windows(2) {
                assert!(
                    w[0].sort_key() < w[1].sort_key(),
                    "page {p} is out of order or has a duplicate"
                );
            }
            let named: std::collections::BTreeSet<(u128, u32, u32, u32)> = refs
                .iter()
                .map(|t| (t.texture().uuid().as_u128(), t.mip, t.x, t.y))
                .collect();
            // Independently: walk the page's own meshlets and place each uv.
            for &g in bytemuck::cast_slice::<u8, u32>(s.indices) {
                let ml = &mesh.meshlets[g as usize];
                let lo = ml.vertex_offset as usize;
                for &vi in &mesh.meshlet_vertices[lo..lo + ml.vertex_count as usize] {
                    let uv = mesh.vertices[vi as usize].uv;
                    for tex in &set.textures {
                        let mip = tile_mip_for_lod(e.lod, tex.mips.len() as u32);
                        let (nx, ny) = tex.mips[mip as usize];
                        let tx = ((uv[0].rem_euclid(1.0) * nx as f32) as u32).min(nx - 1);
                        let ty = ((uv[1].rem_euclid(1.0) * ny as f32) as u32).min(ny - 1);
                        assert!(
                            named.contains(&(tex.id.uuid().as_u128(), mip, tx, ty)),
                            "page {p} samples tile ({tx}, {ty}) of mip {mip} and does not name it"
                        );
                    }
                }
            }
        }
        assert!(total > r.pages().len(), "the pairing is not one tile a page");
    }

    /// No pairing, no section — and the bytes are the same as a build that never
    /// heard of textures, so a mesh with no materials pays nothing.
    #[test]
    fn an_unpaired_build_carries_no_tiles_and_is_byte_identical() {
        let m = dense_mesh(16);
        let a = build_vgeom_asset(&m, &ClusterTextureSet::none()).expect("build");
        let b = build_vgeom_asset(&m, &ClusterTextureSet::default()).expect("build");
        assert_eq!(a.as_bytes(), b.as_bytes());
        for e in a.reader().pages() {
            assert_eq!(e.tile_count, 0);
        }
    }

    /// The pairing is part of the cook's byte determinism, not beside it.
    #[test]
    fn two_paired_builds_are_byte_identical_and_a_third_texture_changes_them() {
        let m = dense_mesh(16);
        let set = ClusterTextureSet {
            textures: vec![tex_2048(1)],
        };
        let a = build_vgeom_asset(&m, &set).expect("build");
        let b = build_vgeom_asset(&m, &set).expect("build");
        assert_eq!(a.as_bytes(), b.as_bytes(), "two builds of one pairing");
        let more = ClusterTextureSet {
            textures: vec![tex_2048(1), tex_2048(2)],
        };
        let c = build_vgeom_asset(&m, &more).expect("build");
        assert_ne!(a.as_bytes(), c.as_bytes(), "a second texture is visible");
    }

    /// A tile reference survives the two conversions it exists to make possible.
    #[test]
    fn a_tile_ref_round_trips_its_asset_id_and_its_coordinate() {
        let id = AssetId(uuid::Uuid::from_u128(
            0x0123_4567_89ab_cdef_fedc_ba98_7654_3210,
        ));
        let r = ClusterTileRef::new(id, 3, 11, 7);
        assert_eq!(r.texture(), id);
        assert_eq!(r.coord(), inf_vt::TileCoord::new(3, 11, 7));
        assert_eq!(r.pad, 0);
    }

    // ── the v2 lift ─────────────────────────────────────────────────────────

    /// The **frozen v2 fixture**: a paged image written by the P18.2 writer, at
    /// the commit before this batch, in a clean worktree. It pins BYTES, not a
    /// generator — the current writer emits v3 and cannot reproduce these, which
    /// is exactly why the file is committed. Never re-bless it from the live
    /// writer; that would turn this gate into a tautology.
    ///
    /// What it proves: every pack cooked between P18.2 and P28.1 still opens, its
    /// 32-byte vertex records widen to 36 correctly, and the lift invents neither
    /// a tangent nor a pairing.
    #[test]
    fn loads_the_frozen_v2_fixture() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/v2_dense12.inf_vmesh");
        let bytes = std::fs::read(&path).expect("committed v2 fixture present");
        assert!(is_v2(&bytes), "the fixture is a paged image");
        assert_eq!(container_version(&bytes), Some(2), "and it is version 2");
        assert!(!is_current(&bytes), "so it is not readable without a lift");

        // The v2 reader still parses it directly, at the v2 vertex stride.
        let direct = VgeomAssetReader::new(bytes.as_slice()).expect("v2 parses");
        assert_eq!(direct.header().schema_version, 2);
        assert!(direct.pages().len() >= 2);
        for e in direct.pages() {
            assert_eq!(e.tile_count, 0, "a v2 page has no pairing");
            assert_eq!(e.tiles_off, 0);
        }
        let v2_mesh = direct.to_mesh().expect("v2 materializes");
        assert!(
            v2_mesh
                .vertices
                .iter()
                .all(|v| v.tangent == crate::model::NO_TANGENT),
            "a v2 payload has no tangents, and the widening must not invent any"
        );

        // And the lift produces a current image with the same geometry.
        let src = VgeomSource::from_payload(bytes).expect("the v2 lift works");
        let lifted = src.to_mesh().expect("materialize the lifted image");
        assert_eq!(lifted.meshlets, v2_mesh.meshlets);
        assert_eq!(lifted.meshlet_triangles, v2_mesh.meshlet_triangles);
        assert_eq!(lifted.levels, v2_mesh.levels);
        assert_eq!(lifted.groups, v2_mesh.groups);
        assert_eq!(lifted.vertices, v2_mesh.vertices);
        for e in src.pages() {
            assert_eq!(e.tile_count, 0, "a lifted asset has no pairing to invent");
        }
    }

    /// A v2 directory that carries v3 fields is a doctored file, not a lenient
    /// one — the version says what the lanes mean and the parse holds it to that.
    #[test]
    fn a_v2_image_claiming_a_tiles_section_is_refused() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/v2_dense12.inf_vmesh");
        let mut bytes = std::fs::read(&path).expect("fixture");
        // `tile_count` of page 0 lives at directory offset 40.
        let at = HEADER_LEN as usize + 40;
        bytes[at..at + 4].copy_from_slice(&1u32.to_le_bytes());
        assert!(matches!(
            VgeomAssetReader::new(bytes.as_slice()),
            Err(VgeomAssetError::Malformed(_))
        ));
    }

    #[test]
    fn header_reports_the_draw_shape_without_paging() {
        let m = dense_mesh(20);
        let src = VgeomSource::from_mesh(&m).unwrap();
        let want = m.meshlets.iter().map(|x| x.triangle_count).max().unwrap();
        assert_eq!(src.max_tri(), want);
        assert_eq!(src.bounds(), (m.center, m.radius));
        assert!(src.total_resident_bytes() > 0);
        assert_eq!(src.meshlet_count(), m.meshlets.len() as u32);
    }

    /// An empty mesh (no geometry) still produces a valid, openable image.
    #[test]
    fn empty_mesh_round_trips() {
        let m = VgeomMesh {
            schema_version: VgeomMesh::CURRENT_VERSION,
            vertices: Vec::new(),
            meshlets: Vec::new(),
            meshlet_vertices: Vec::new(),
            meshlet_triangles: Vec::new(),
            groups: Vec::new(),
            levels: Vec::new(),
            center: [0.0; 3],
            radius: 0.0,
        };
        let src = VgeomSource::from_mesh(&m).unwrap();
        assert_eq!(src.pages().len(), 0);
        assert_eq!(src.meshlet_count(), 0);
        assert_eq!(src.to_mesh().unwrap().meshlets.len(), 0);
    }
}
