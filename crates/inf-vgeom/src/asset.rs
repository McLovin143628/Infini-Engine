//! The `.inf_vmesh` **streaming asset** (P18.2): the meshlet DAG leaves its
//! single bincode blob for a random-access, page-at-a-time container.
//!
//! ```text
//! ┌ header (128 B, little-endian) ────────────────────────────────────┐
//! │  magic         [u8; 8]   b"INFVMSH\0"                             │
//! │  schema_ver    u32       VMESH_ASSET_SCHEMA_VERSION (2)           │
//! │  page_count    u32       streaming pages (coarsest/root page 0)   │
//! │  meshlet_count u32       meshlets across all pages                │
//! │  vertex_count  u32       vertex records across all pages          │
//! │  max_tri       u32       max triangle_count (the indirect draw's) │
//! │  max_lod       u32       coarsest LOD level present               │
//! │  center        [f32; 3] · radius f32                              │
//! │  page_base     u64       absolute offset of the section area      │
//! │  groups_off    u64 · groups_len u64   the DAG groups (bincode)    │
//! │  total_len     u64       payload length (a self-check)            │
//! │  reserved      [u8; 40]  zeros (room for v3 without a re-length)  │
//! ├ page directory (page_count × 96 B, COARSEST FIRST) ───────────────┤
//! │  page u32 · meshlet_count u32 · vertex_start u32 · vertex_count   │
//! │  u32 · mlvert_count u32 · mltri_count u32 · floor_lod u32 ·       │
//! │  lod u32 · max_parent_error f32 · min_error f32 · pad[2] ·        │
//! │  indices_off · meshlets_off · mlverts_off · mltris_off ·          │
//! │  vertices_off (u64 each) · pad u64                                │
//! ├ page sections (indices · meshlets · mlverts · mltris · vertices) ─┤
//! │  … each 16-byte aligned, zero-padded up to the next boundary …    │
//! ├ groups section (bincode `Vec<Group>`) ────────────────────────────┤
//! └───────────────────────────────────────────────────────────────────┘
//! ```
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
/// v1 is the bare bincode [`VgeomMesh`] (no magic) and keeps loading forever; v2
/// is the paged image this module writes. This versions the *container* and is
/// independent of [`VgeomMesh::CURRENT_VERSION`], which versions the in-memory
/// record and is unchanged by this batch.
pub const VMESH_ASSET_SCHEMA_VERSION: u32 = 2;

/// Sections start on multiples of this many bytes — the same constant, and the
/// same reasoning, as [`inf_asset::BLOB_ALIGN`] and `.inf_terrain`'s `TILE_ALIGN`.
pub const SECTION_ALIGN: u64 = 16;

/// Bytes of the fixed header.
pub const HEADER_LEN: u64 = 128;

/// Bytes of one page-directory entry.
pub const PAGE_ENTRY_LEN: u64 = 96;

/// Bytes of one on-disk meshlet record ([`MeshletRec`]).
pub const MESHLET_REC_LEN: usize = 64;

/// Bytes of one on-disk vertex record ([`VgeomVertex`]).
pub const VERTEX_REC_LEN: usize = 32;

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
    /// `u32` global meshlet index per meshlet — how the remap table is built.
    pub indices_off: u64,
    pub meshlets_off: u64,
    pub mlverts_off: u64,
    pub mltris_off: u64,
    pub vertices_off: u64,
}

impl VgeomPageEntry {
    /// Bytes this page occupies in the four GPU pools if it is made resident —
    /// the quantity the VRAM budget is spent in. The `indices` section is CPU-side
    /// bookkeeping and is deliberately **not** counted: it is never uploaded.
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
}

// ── builder ─────────────────────────────────────────────────────────────────

/// Lay a [`VgeomMesh`] out as a v2 paged image.
///
/// Pure and byte-deterministic: the page partition, the vertex permutation and the
/// section order are all functions of the mesh alone, and padding is deterministic
/// zeros — so two builds of one mesh are byte-identical (the cook's guarantee).
pub fn build_vgeom_asset(mesh: &VgeomMesh) -> Result<VgeomAssetImage> {
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
        max_parent_error: f32,
        min_error: f32,
    }
    let mut blobs: Vec<PageBlob> = Vec::with_capacity(page_count);
    for members in &page_members {
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
        blobs.push(PageBlob {
            indices,
            meshlets,
            mlverts: bytemuck::cast_slice(&mlverts).to_vec(),
            mltris,
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
    let mut offsets: Vec<[u64; 5]> = Vec::with_capacity(page_count);
    for (p, b) in blobs.iter().enumerate() {
        let mut here = [0u64; 5];
        for (slot, len) in [
            b.indices.len() as u64,
            b.meshlets.len() as u64,
            b.mlverts.len() as u64,
            b.mltris.len() as u64,
            page_vertex_count[p] as u64 * VERTEX_REC_LEN as u64,
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
        out.extend_from_slice(&[0u8; 8]); // pad
        for v in offsets[p] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&[0u8; 8]); // pad
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
    }

    out.resize(groups_off as usize, 0);
    out.extend_from_slice(&groups_bytes);
    out.resize(total_len as usize, 0);

    Ok(VgeomAssetImage { bytes: out })
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
                e.vertex_count as u64 * VERTEX_REC_LEN as u64,
            ),
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
            vertices.extend_from_slice(bytemuck::cast_slice::<u8, VgeomVertex>(s.vertices));
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
    for (count, stride, what) in [
        (header.vertex_count, VERTEX_REC_LEN as u64, "vertices"),
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
            indices_off: u64_at(b + 48),
            meshlets_off: u64_at(b + 56),
            mlverts_off: u64_at(b + 64),
            mltris_off: u64_at(b + 72),
            vertices_off: u64_at(b + 80),
        };
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
            (
                e.vertices_off,
                e.vertex_count as u64 * VERTEX_REC_LEN as u64,
                "vertices",
            ),
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
        if !is_v2(payload.as_ref()) {
            // A v1 (bincode) entry has to be lifted into a v2 image, which means
            // owning bytes either way.
            let mesh = decode_v1(payload.as_ref()).map_err(|e| format!("vmesh {guid}: {e}"))?;
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
    /// **v1** bincode payload if that is what it is.
    pub fn from_payload(bytes: Vec<u8>) -> std::result::Result<Self, String> {
        if !is_v2(&bytes) {
            let mesh = decode_v1(&bytes)?;
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

    /// Lay an in-memory [`VgeomMesh`] out as a v2 image and index it.
    ///
    /// This is what makes "one reader for everything" true: an editor build, a
    /// test fixture and a lifted v1 payload all reach the renderer through the
    /// **same** paged path a cooked pack does, so the streaming code has no second
    /// shape to be right about.
    pub fn from_mesh(mesh: &VgeomMesh) -> std::result::Result<Self, String> {
        let image = build_vgeom_asset(mesh).map_err(|e| e.to_string())?;
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

/// Whether `bytes` look like a v2 paged image (as opposed to a v1 bincode
/// `VgeomMesh`, whose first four bytes are `schema_version = 1`).
pub fn is_v2(bytes: &[u8]) -> bool {
    bytes.len() >= 8 && bytes[0..8] == VMESH_ASSET_MAGIC
}

/// Decode a **v1** payload: the bare bincode [`VgeomMesh`] every pack cooked
/// before P18.2 carries.
fn decode_v1(bytes: &[u8]) -> std::result::Result<VgeomMesh, String> {
    inf_asset::decode::<VgeomMesh>(bytes).map_err(|e| format!("decode v1 vmesh: {e}"))
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
        let image = build_vgeom_asset(&m).expect("build");
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
        let a = build_vgeom_asset(&m).unwrap().into_bytes();
        let b = build_vgeom_asset(&m).unwrap().into_bytes();
        assert_eq!(a, b, "two builds of one mesh are byte-identical");
    }

    #[test]
    fn sections_are_aligned_and_in_bounds() {
        let m = dense_mesh(24);
        let image = build_vgeom_asset(&m).unwrap();
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
        let image = build_vgeom_asset(&m).unwrap();
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
        let image = build_vgeom_asset(&m).unwrap();
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
        let direct: VgeomMesh = inf_asset::decode(&bytes).expect("v1 decodes directly");
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
        let mut bytes = build_vgeom_asset(&m).unwrap().into_bytes();
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
        let mut bytes = build_vgeom_asset(&m).unwrap().into_bytes();
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
        let good = build_vgeom_asset(&m).unwrap().into_bytes();
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
        let good = build_vgeom_asset(&m).unwrap().into_bytes();
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
