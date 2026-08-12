//! Asset thumbnails: a headless-wgpu renderer + a content-hash disk cache.
//!
//! Thumbnails are keyed by the asset's payload content hash, so an unchanged
//! asset renders once and is served from disk forever after (and a re-import
//! that changes the bytes gets a fresh thumbnail automatically). Type-specific
//! renderers: meshes get a shaded 3/4 view of the real geometry, materials a lit
//! sphere, textures a flat (CPU-decoded, letterboxed) preview.
//!
//! The GPU context is created lazily and may be absent (headless CI without an
//! adapter): mesh/material thumbnails then fail gracefully and the UI falls back
//! to a type icon; texture thumbnails never need a GPU.

mod scene_render;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use inf_asset::{AssetId, AssetKind};
use inf_material::{MaterialAsset, TextureAsset};
use inf_mesh::{MeshAsset, MeshVertex};
use inf_render::GpuContext;

use crate::assets::AssetProject;

/// Default thumbnail edge length (square).
pub const THUMB_SIZE: u32 = 128;

pub use scene_render::{PreviewSession, PreviewView};

/// Renders asset thumbnails. Holds a lazily-created, possibly-absent GPU context.
///
/// It also owns the live [`PreviewSession`] (P23.2a) that
/// [`render_material_preview`](Thumbnailer::render_material_preview) draws
/// through, so the material editor's preview keeps its target, buffers and
/// compiled pipelines between compiles instead of rebuilding them per keystroke.
pub struct Thumbnailer {
    size: u32,
    gpu: GpuState,
    /// Created on first use and then kept for the process's life. A size change
    /// **resizes** it rather than replacing it, so the compiled pipelines
    /// survive (P23.2a audit — M5): `material_compile` asks for 256 and a Model
    /// Editor panel will ask for 512, and rebuilding would have had the two
    /// consumers evict each other's programs on every alternation.
    preview: Option<PreviewSession>,
}

enum GpuState {
    Uninit,
    Ready(Box<GpuContext>),
    Unavailable,
}

impl Default for Thumbnailer {
    fn default() -> Self {
        Self::new(THUMB_SIZE)
    }
}

impl Thumbnailer {
    pub fn new(size: u32) -> Self {
        Self {
            size,
            gpu: GpuState::Uninit,
            preview: None,
        }
    }

    /// Bring the GPU context up if it has not been tried yet. Split out of
    /// [`gpu`](Self::gpu) so a caller can initialize and then borrow `self.gpu`
    /// and another field disjointly (the preview session needs both).
    fn ensure_gpu(&mut self) {
        if matches!(self.gpu, GpuState::Uninit) {
            self.gpu = match GpuContext::headless() {
                Ok(ctx) => {
                    // **This device is LIVE in the editor** (P23.2a audit — M3).
                    //
                    // `GpuContext::headless` deliberately keeps wgpu's fatal
                    // default handler, because its other callers are CI: a
                    // golden or a validation regression must abort loudly
                    // rather than degrade. But the editor's `MaterialState`
                    // holds a `Thumbnailer` for the whole session and drives it
                    // from the material panel on every graph edit — so on this
                    // one instance the fatal default means a bad draw takes the
                    // EDITOR down, with the author's unsaved level in it.
                    //
                    // The composed-module validation in `PreviewSession` catches
                    // the failure we know about; this is the net under the ones
                    // we do not. Errors are counted and rate-limited-logged.
                    ctx.install_lenient_error_handler();
                    GpuState::Ready(Box::new(ctx))
                }
                Err(e) => {
                    tracing::info!("thumbnailer: no GPU adapter ({e}); mesh/material previews off");
                    GpuState::Unavailable
                }
            };
        }
    }

    /// Lazily obtain the GPU context; `None` if this machine has no adapter.
    fn gpu(&mut self) -> Option<&GpuContext> {
        self.ensure_gpu();
        match &self.gpu {
            GpuState::Ready(ctx) => Some(ctx),
            _ => None,
        }
    }

    /// Render an asset to a square RGBA8 image. Returns `None` when the asset
    /// kind has no visual preview or the GPU is unavailable for a 3D kind.
    ///
    /// The `project` lock is taken only briefly to resolve *what* to draw (kind,
    /// on-disk payload path, and — for meshes — a base-color hint from the first
    /// material dependency); the payload file read + GPU render then run with the
    /// lock released, so a thumbnail render never blocks other asset commands on
    /// the project mutex.
    pub fn render_rgba(
        &mut self,
        project: &Arc<Mutex<AssetProject>>,
        id: AssetId,
    ) -> Option<Vec<u8>> {
        enum Draw {
            Texture(PathBuf),
            Mesh(PathBuf, [f32; 4]),
            Material(PathBuf),
        }
        // Resolve the draw job under a short project lock, then release it.
        let draw = {
            let proj = project.lock().ok()?;
            let entry = proj.db().get(id)?;
            match entry.kind() {
                AssetKind::Texture => Draw::Texture(entry.path.clone()),
                AssetKind::Mesh => {
                    let base = mesh_base_color(&proj, entry.sidecar.dependencies.as_slice());
                    Draw::Mesh(entry.path.clone(), base)
                }
                AssetKind::Material => Draw::Material(entry.path.clone()),
                _ => return None,
            }
        };
        match draw {
            Draw::Texture(path) => {
                // Not `load_payload`: since P26.1 a `.inf_tex` is a tiled image
                // whose bytes are not bincode. `from_payload` sniffs the magic
                // and reconstructs the same whole-levels record from either
                // version, so everything below this line is unchanged.
                let tex = TextureAsset::from_payload(&std::fs::read(&path).ok()?).ok()?;
                Some(texture_thumbnail(&tex, self.size))
            }
            Draw::Mesh(path, base) => {
                let mesh: MeshAsset = load_payload(&path).ok()?;
                let (verts, indices) = combined_geometry(&mesh);
                let bounds = mesh.bounds;
                let size = self.size;
                let gpu = self.gpu()?;
                scene_render::render_mesh(gpu, size, &verts, &indices, bounds, base).ok()
            }
            Draw::Material(path) => {
                let mat: MaterialAsset = load_payload(&path).ok()?;
                let size = self.size;
                let gpu = self.gpu()?;
                scene_render::render_sphere(gpu, size, mat.base_color).ok()
            }
        }
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    /// Render a material graph's generated `surface_wgsl` on a lit preview
    /// sphere (P7.2.3). `None` if no GPU adapter is available. `tex_count`
    /// texture slots are bound to white; the surface must be naga-valid.
    ///
    /// Goes through the persistent [`PreviewSession`] (P23.2a): the material
    /// editor recompiles on every graph edit, so the target, depth buffer,
    /// sphere buffers and — when the same surface comes back, as it does on an
    /// undo or a parameter revert — the compiled pipeline are all reused. The
    /// image is identical to the pre-session one-shot; only the cost changed.
    ///
    /// Three outcomes, all of them honest (P23.2a audit — M3):
    /// `Ok(Some(rgba))` rendered; `Ok(None)` this machine has no GPU adapter
    /// (the standing degrade-to-icons case); `Err(msg)` the composed preview
    /// shader did not validate, which the panel shows in its diagnostics slot
    /// rather than dying on.
    pub fn render_material_preview(
        &mut self,
        surface_wgsl: &str,
        tex_count: u32,
        size: u32,
    ) -> Result<Option<Vec<u8>>, String> {
        self.render_preview_view(surface_wgsl, tex_count, size, PreviewView::default())
    }

    /// Render an interactive preview frame at an explicit camera (P23.2a) — the
    /// seam a Model Editor panel drives with a mouse-orbit.
    ///
    /// The same session as [`render_material_preview`](Self::render_material_preview),
    /// so a panel that orbits and a compile that re-renders share one set of
    /// compiled pipelines — including **across a size change**, which
    /// `PreviewSession::resize` keeps (the two consumers ask for 512 and 256).
    pub fn render_preview_view(
        &mut self,
        surface_wgsl: &str,
        tex_count: u32,
        size: u32,
        view: PreviewView,
    ) -> Result<Option<Vec<u8>>, String> {
        self.ensure_gpu();
        let GpuState::Ready(gpu) = &self.gpu else {
            return Ok(None);
        };
        match &mut self.preview {
            Some(session) => session.resize(gpu, size),
            slot @ None => *slot = Some(PreviewSession::new(gpu, size)),
        }
        let Some(session) = self.preview.as_mut() else {
            return Ok(None);
        };
        session.render(gpu, surface_wgsl, tex_count, view).map(Some)
    }

    /// Render an arbitrary triangle mesh through the same cached session — how
    /// the Model Editor previews an **edit mesh** (P23.4).
    ///
    /// `stamp` keys the geometry upload (see [`PreviewSession::set_geometry`]):
    /// pass the edit session's journal generation, and a camera orbit re-renders
    /// without touching a vertex buffer.
    ///
    /// Same three honest outcomes as the material preview: rendered, no adapter
    /// on this machine, or a shader that did not validate.
    pub fn render_geometry_view(
        &mut self,
        verts: &[MeshVertex],
        indices: &[u32],
        stamp: u64,
        surface_wgsl: &str,
        size: u32,
        view: PreviewView,
    ) -> Result<Option<Vec<u8>>, String> {
        self.ensure_gpu();
        let GpuState::Ready(gpu) = &self.gpu else {
            return Ok(None);
        };
        match &mut self.preview {
            Some(session) => session.resize(gpu, size),
            slot @ None => *slot = Some(PreviewSession::new(gpu, size)),
        }
        let Some(session) = self.preview.as_mut() else {
            return Ok(None);
        };
        session.set_geometry(gpu, verts, indices, stamp);
        session.render(gpu, surface_wgsl, 0, view).map(Some)
    }

    /// Bake a texture graph's generated `compute_wgsl` (entry `cs_bake`) to a
    /// `size × size` RGBA8 image (P7.3). `None` if no GPU adapter is available.
    pub fn bake_texture(
        &mut self,
        compute_wgsl: &str,
        tex_count: u32,
        size: u32,
    ) -> Option<Vec<u8>> {
        let gpu = self.gpu()?;
        scene_render::bake_texture(gpu, size, compute_wgsl, tex_count).ok()
    }
}

/// A content-hash-keyed on-disk cache of encoded PNG thumbnails.
pub struct ThumbnailCache {
    dir: PathBuf,
}

impl ThumbnailCache {
    /// Open (creating) the cache directory (usually `<project>/.inf/thumbnails`).
    pub fn open(dir: impl Into<PathBuf>) -> std::io::Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    fn path_for(&self, hash_hex: &str) -> PathBuf {
        self.dir.join(format!("{hash_hex}.png"))
    }

    /// Return the cached PNG path for `id`, rendering + caching it on a miss.
    /// Returns `None` if the asset has no preview (or a 3D kind with no GPU).
    ///
    /// Locks `project` only to key by content hash (and, on a miss, for the
    /// render's brief metadata resolve) — never across the GPU render or the
    /// PNG-encode/file-write.
    pub fn get_or_render(
        &self,
        project: &Arc<Mutex<AssetProject>>,
        id: AssetId,
        thumbnailer: &mut Thumbnailer,
    ) -> Option<PathBuf> {
        let hash = {
            let proj = project.lock().ok()?;
            proj.db().get(id)?.sidecar.content_hash.to_hex()
        };
        let path = self.path_for(&hash);
        if path.exists() {
            return Some(path);
        }
        let rgba = thumbnailer.render_rgba(project, id)?;
        let png = encode_png(thumbnailer.size(), &rgba).ok()?;
        std::fs::write(&path, png).ok()?;
        Some(path)
    }

    /// True if a thumbnail is already cached for this asset's current content.
    pub fn is_cached(&self, project: &AssetProject, id: AssetId) -> bool {
        project
            .db()
            .get(id)
            .map(|e| self.path_for(&e.sidecar.content_hash.to_hex()).exists())
            .unwrap_or(false)
    }

    /// Evict orphaned thumbnails: delete every cached PNG whose content hash no
    /// longer matches any live asset (each re-import/edit re-keys the thumbnail
    /// by a fresh hash, orphaning the old one). Called at a non-hot moment — a
    /// project open/rescan — never on a render path.
    ///
    /// Safe against the cache dir not existing (returns 0), and only ever touches
    /// files matching the cache's own `<32-hex>.png` naming scheme, so foreign
    /// files dropped in the directory are never deleted. Returns the number of
    /// orphaned PNGs removed.
    pub fn sweep(&self, project: &AssetProject) -> usize {
        use std::collections::HashSet;
        let live: HashSet<String> = project
            .db()
            .iter()
            .map(|e| e.content_hash().to_hex())
            .collect();
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return 0; // cache dir absent → nothing to sweep
        };
        let mut removed = 0;
        for entry in entries.flatten() {
            let path = entry.path();
            // Only our own `<hash>.png` files are candidates; skip everything else.
            let Some(hash) = cache_hash_stem(&path) else {
                continue;
            };
            if !live.contains(&hash) && std::fs::remove_file(&path).is_ok() {
                removed += 1;
            }
        }
        removed
    }
}

/// The content-hash stem of a path that matches the cache's own
/// `<32-lowercase-hex>.png` naming scheme (see [`ThumbnailCache::path_for`]), or
/// `None` for any other file — the guard that keeps [`ThumbnailCache::sweep`]
/// from deleting foreign files.
fn cache_hash_stem(path: &Path) -> Option<String> {
    if path.extension().and_then(|e| e.to_str()) != Some("png") {
        return None;
    }
    let stem = path.file_stem().and_then(|s| s.to_str())?;
    let is_lower_hex = stem.len() == 32
        && stem
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
    is_lower_hex.then(|| stem.to_string())
}

// ── payload loading ───────────────────────────────────────────────────────

fn load_payload<T: inf_asset::AssetPayload>(path: &Path) -> inf_asset::Result<T> {
    let bytes = std::fs::read(path)?;
    inf_asset::decode(&bytes)
}

/// Concatenate a mesh's submeshes into one vertex + index buffer (offsetting
/// indices), for a single draw.
fn combined_geometry(mesh: &MeshAsset) -> (Vec<MeshVertex>, Vec<u32>) {
    let mut verts = Vec::new();
    let mut indices = Vec::new();
    for sm in &mesh.submeshes {
        let base = verts.len() as u32;
        verts.extend_from_slice(&sm.vertices);
        indices.extend(sm.indices.iter().map(|&i| i + base));
    }
    (verts, indices)
}

/// Look up the base color of a mesh's first material dependency (for a hint of
/// its real color in the preview), defaulting to a neutral grey.
fn mesh_base_color(project: &AssetProject, deps: &[AssetId]) -> [f32; 4] {
    for &dep in deps {
        if let Some(e) = project.db().get(dep) {
            if e.kind() == AssetKind::Material {
                if let Ok(mat) = load_payload::<MaterialAsset>(&e.path) {
                    return mat.base_color;
                }
            }
        }
    }
    [0.72, 0.72, 0.74, 1.0]
}

// ── texture (CPU) thumbnail ────────────────────────────────────────────────

/// A flat, letterboxed square preview of a texture that is **not in a project
/// yet** — the same door [`Thumbnailer::render_rgba`] uses for an `.inf_tex`,
/// reached with a payload in hand instead of an [`AssetId`].
///
/// P25.4's capture wizard shows the baked base colour beside the offscreen
/// geometry preview *before* the scan is imported, so there is no asset to ask
/// for. It needs no GPU, which is the other reason it is separate: the atlas is
/// visible on a machine where the geometry preview is not.
pub fn texture_preview_rgba(tex: &TextureAsset, size: u32) -> Vec<u8> {
    texture_thumbnail(tex, size)
}

/// A flat, letterboxed square preview of a texture (decodes the smallest mip
/// that still covers the thumbnail, for speed).
fn texture_thumbnail(tex: &TextureAsset, size: u32) -> Vec<u8> {
    // Pick the smallest mip whose max dimension is >= size (or mip 0).
    let level = (0..tex.mip_count())
        .rev()
        .find(|&l| tex.mips[l].width.max(tex.mips[l].height) >= size)
        .unwrap_or(0);
    let mip = &tex.mips[level];
    let rgba = tex.level_rgba8(level).unwrap_or_else(|| vec![0; 4]);
    resize_letterbox(&rgba, mip.width.max(1), mip.height.max(1), size)
}

/// Resize an RGBA8 image into a `size×size` square, preserving aspect (nearest
/// sampling), filling the letterbox with the studio-grey backdrop.
fn resize_letterbox(src: &[u8], sw: u32, sh: u32, size: u32) -> Vec<u8> {
    let mut out = vec![0u8; (size * size * 4) as usize];
    // Backdrop.
    for px in out.chunks_exact_mut(4) {
        px.copy_from_slice(&[31, 31, 36, 255]);
    }
    let scale = (size as f32 / sw as f32).min(size as f32 / sh as f32);
    let dw = ((sw as f32 * scale).round() as u32).max(1);
    let dh = ((sh as f32 * scale).round() as u32).max(1);
    let ox = (size - dw) / 2;
    let oy = (size - dh) / 2;
    for y in 0..dh {
        let syf = (y as f32 + 0.5) / scale;
        let sy = (syf as u32).min(sh - 1);
        for x in 0..dw {
            let sxf = (x as f32 + 0.5) / scale;
            let sx = (sxf as u32).min(sw - 1);
            let si = ((sy * sw + sx) * 4) as usize;
            let di = (((oy + y) * size + (ox + x)) * 4) as usize;
            if si + 4 <= src.len() {
                out[di..di + 4].copy_from_slice(&src[si..si + 4]);
            }
        }
    }
    out
}

/// PNG-encode a square RGBA8 image.
///
/// Default deflate: a thumbnail is encoded ONCE and lives on disk under its
/// content hash for ever, so the bytes are worth compressing hard.
pub fn encode_png(size: u32, rgba: &[u8]) -> Result<Vec<u8>, String> {
    encode_png_with(
        size,
        rgba,
        png::Compression::default(),
        png::Filter::default(),
    )
}

/// PNG-encode for an **interactive** preview (P23.2a) — same pixels, cheaper.
///
/// The P23.2a measurement found the offscreen preview's cost is not the render:
/// at 512² a warm re-render is ~0.34 ms and the default-compression PNG encode
/// is ~22.7 ms, i.e. **98% of the frame is deflate**. Nothing downstream keeps
/// this image — it is base64'd into a data URL, shown, and replaced by the next
/// orbit frame — so paying for the last few percent of size is paying for
/// nothing at a cost that decides whether the panel is interactive.
///
/// PNG is lossless at every level, so this is byte-identical *as an image*; only
/// the file is larger. Never use it for the disk cache.
pub fn encode_png_fast(size: u32, rgba: &[u8]) -> Result<Vec<u8>, String> {
    // `NoFilter` as well as `Fast`: adaptive filtering walks every row five
    // times to pick a predictor, which is most of what is being paid for here.
    encode_png_with(size, rgba, png::Compression::Fast, png::Filter::NoFilter)
}

fn encode_png_with(
    size: u32,
    rgba: &[u8],
    compression: png::Compression,
    filter: png::Filter,
) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut buf, size, size);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(compression);
        encoder.set_filter(filter);
        let mut writer = encoder.write_header().map_err(|e| e.to_string())?;
        writer.write_image_data(rgba).map_err(|e| e.to_string())?;
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_material::{texture_from_rgba8, TextureImportSettings};

    fn project_with_texture() -> (tempfile::TempDir, Arc<Mutex<AssetProject>>, AssetId) {
        let dir = tempfile::tempdir().unwrap();
        let mut proj = AssetProject::open(dir.path()).unwrap();
        let d = proj.content_dir("tex").unwrap();
        // A 16×16 red/blue checker.
        let mut rgba = Vec::new();
        for y in 0..16 {
            for x in 0..16 {
                if (x + y) % 2 == 0 {
                    rgba.extend_from_slice(&[220, 40, 40, 255]);
                } else {
                    rgba.extend_from_slice(&[40, 40, 220, 255]);
                }
            }
        }
        let tex = texture_from_rgba8(rgba, 16, 16, TextureImportSettings::default()).unwrap();
        let id = proj
            .write_asset(&d, "Checker", &tex, None, vec![], None)
            .unwrap();
        (dir, Arc::new(Mutex::new(proj)), id)
    }

    /// **The v1-compatibility measurement** (P26.1). The thumbnailer is the
    /// oldest consumer of a `.inf_tex` and the one an author sees; it must not
    /// be able to tell which container version it was handed.
    ///
    /// Asserted on the PIXELS, byte for byte, from two projects holding the two
    /// containers of the *same* source — not on "both produce something", which
    /// two different images also satisfy. The fixture is asymmetric (four
    /// distinct corner quadrants) so a mirrored or transposed reconstruction
    /// cannot pass, and it is 132×132 so mip 0 spans four tiles rather than one.
    #[test]
    fn the_thumbnail_is_byte_identical_for_v1_and_its_v2_re_encode() {
        const N: u32 = 132;
        let mut rgba = Vec::with_capacity((N * N * 4) as usize);
        for y in 0..N {
            for x in 0..N {
                let c: [u8; 3] = match (x * 2 < N, y * 2 < N) {
                    (true, true) => [220, 20, 20],
                    (false, true) => [20, 200, 20],
                    (true, false) => [20, 20, 210],
                    (false, false) => [230, 210, 30],
                };
                rgba.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
        }
        let s = TextureImportSettings::default();

        let render = |write: &dyn Fn(&mut AssetProject, &std::path::Path) -> AssetId| {
            let dir = tempfile::tempdir().unwrap();
            let mut proj = AssetProject::open(dir.path()).unwrap();
            let d = proj.content_dir("tex").unwrap();
            let id = write(&mut proj, &d);
            let proj = Arc::new(Mutex::new(proj));
            let mut thumb = Thumbnailer::new(64);
            let out = thumb.render_rgba(&proj, id).expect("texture thumbnail");
            (dir, out)
        };

        let r1 = rgba.clone();
        let (_d1, v1) = render(&move |proj, d| {
            let tex = texture_from_rgba8(r1.clone(), N, N, s).unwrap();
            proj.write_asset(d, "Corners", &tex, None, vec![], None)
                .unwrap()
        });
        let r2 = rgba.clone();
        let (_d2, v2) = render(&move |proj, d| {
            let img = inf_material::build_tiled_texture(r2.clone(), N, N, s).unwrap();
            proj.write_tiled_texture(d, "Corners", &img, None, None)
                .unwrap()
        });

        assert_eq!(v1.len(), 64 * 64 * 4);
        assert_eq!(v1, v2, "the v2 thumbnail differs from the v1 thumbnail");
        // Anti-vacuity: the thumbnail is a real image, not a uniform backdrop —
        // two equal *blank* buffers would satisfy the assertion above.
        let distinct: std::collections::BTreeSet<[u8; 4]> = v1
            .chunks_exact(4)
            .map(|p| [p[0], p[1], p[2], p[3]])
            .collect();
        assert!(
            distinct.len() >= 4,
            "only {} distinct colours",
            distinct.len()
        );
    }

    #[test]
    fn texture_thumbnail_renders_without_a_gpu() {
        let (_dir, proj, id) = project_with_texture();
        let mut thumb = Thumbnailer::new(64);
        let rgba = thumb.render_rgba(&proj, id).expect("texture thumbnail");
        assert_eq!(rgba.len(), 64 * 64 * 4);
        // The center pixels are one of the checker colors (not the backdrop).
        let c = (((32 * 64) + 32) * 4) as usize;
        assert!(rgba[c] > 30 || rgba[c + 2] > 30);
    }

    #[test]
    fn cache_stores_and_reuses_by_content_hash() {
        let (dir, proj, id) = project_with_texture();
        let cache = ThumbnailCache::open(dir.path().join(".inf/thumbnails")).unwrap();
        let mut thumb = Thumbnailer::new(64);
        assert!(!cache.is_cached(&proj.lock().unwrap(), id));
        let path = cache.get_or_render(&proj, id, &mut thumb).unwrap();
        assert!(path.exists(), "PNG written");
        assert!(cache.is_cached(&proj.lock().unwrap(), id));
        // Second call is a cache hit (same path).
        let again = cache.get_or_render(&proj, id, &mut thumb).unwrap();
        assert_eq!(path, again);
    }

    #[test]
    fn sweep_removes_only_orphaned_thumbnails() {
        let (dir, proj, id) = project_with_texture();
        let cache = ThumbnailCache::open(dir.path().join(".inf/thumbnails")).unwrap();
        let mut thumb = Thumbnailer::new(64);

        // Cache a live thumbnail (the texture renders without a GPU).
        let live_path = cache.get_or_render(&proj, id, &mut thumb).unwrap();
        // Plant an orphan whose hash matches no live asset, and a foreign file.
        let orphan = cache.dir.join(format!("{:032x}.png", 0xdead_beef_u128));
        std::fs::write(&orphan, b"stale png").unwrap();
        let foreign = cache.dir.join("notes.txt");
        std::fs::write(&foreign, b"keep me").unwrap();

        let removed = cache.sweep(&proj.lock().unwrap());

        assert_eq!(removed, 1, "only the orphan is swept");
        assert!(live_path.exists(), "the live thumbnail survives");
        assert!(!orphan.exists(), "the orphaned PNG is deleted");
        assert!(foreign.exists(), "a non-cache file is never touched");
    }

    #[test]
    fn sweep_is_safe_when_the_cache_dir_is_absent() {
        let (_dir, proj, _id) = project_with_texture();
        // A directory that was never created (bypass `open`, which would make it).
        let cache = ThumbnailCache {
            dir: std::env::temp_dir().join("inf-thumb-cache-does-not-exist-xyz"),
        };
        assert_eq!(
            cache.sweep(&proj.lock().unwrap()),
            0,
            "missing cache dir sweeps nothing"
        );
    }

    #[test]
    fn png_encodes_to_a_valid_signature() {
        let rgba = vec![128u8; 8 * 8 * 4];
        let png = encode_png(8, &rgba).unwrap();
        assert_eq!(&png[1..4], b"PNG", "PNG magic present");
    }

    /// Decode a square RGBA8 PNG back to tightly-packed rows.
    fn decode_png(bytes: &[u8]) -> Vec<u8> {
        let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
        let mut reader = decoder.read_info().expect("valid PNG header");
        let mut buf = vec![0u8; reader.output_buffer_size().expect("bounded frame")];
        let info = reader.next_frame(&mut buf).expect("valid PNG frame");
        assert_eq!(info.color_type, png::ColorType::Rgba);
        assert_eq!(info.bit_depth, png::BitDepth::Eight);
        buf.truncate(info.buffer_size());
        buf
    }

    /// **`encode_png_fast` is the same IMAGE, only a bigger file** (P23.2a
    /// audit).
    ///
    /// The interactive encoder differs from the archival one only in deflate
    /// level and row filter, both of which PNG defines as lossless — so the
    /// claim "same pixels" is true by construction. That is exactly the kind of
    /// by-construction argument this codebase has been burned by, and it costs
    /// three lines to check instead: decode both and compare to the input.
    ///
    /// A gradient, not a flat fill: a constant image survives any filter/level
    /// combination trivially, so it would pass even if `NoFilter` were wired to
    /// something that mangled row prediction.
    #[test]
    fn the_fast_encoder_round_trips_to_the_same_pixels() {
        let size = 16u32;
        let mut rgba = Vec::with_capacity((size * size * 4) as usize);
        for y in 0..size {
            for x in 0..size {
                rgba.extend_from_slice(&[(x * 16) as u8, (y * 16) as u8, 64, 255]);
            }
        }

        let archival = encode_png(size, &rgba).expect("default encode");
        let fast = encode_png_fast(size, &rgba).expect("fast encode");
        assert_eq!(&fast[1..4], b"PNG", "the fast encoder still emits a PNG");

        assert_eq!(
            decode_png(&archival),
            rgba,
            "the archival encoder is lossless"
        );
        assert_eq!(
            decode_png(&fast),
            rgba,
            "the FAST encoder changed the pixels — it may only change the file"
        );

        // …and it really is trading size for speed rather than being the same
        // call twice, which is what would make the claim above vacuous.
        assert!(
            fast.len() > archival.len(),
            "the fast encode ({} B) is not larger than the archival one ({} B) — \
             the two settings are not distinct, so this test proves nothing about \
             the encoder the Model Editor will use",
            fast.len(),
            archival.len()
        );
    }

    /// Mesh thumbnail exercises the GPU path — skipped where no adapter exists.
    #[test]
    fn mesh_thumbnail_when_gpu_available() {
        let dir = tempfile::tempdir().unwrap();
        let mut proj = AssetProject::open(dir.path()).unwrap();
        let d = proj.content_dir("mesh").unwrap();
        // A unit quad mesh.
        let v = |x: f32, y: f32| MeshVertex {
            position: [x, y, 0.0],
            normal: [0.0, 0.0, 1.0],
            ..Default::default()
        };
        let sm = inf_mesh::SubMesh {
            name: "q".into(),
            vertices: vec![v(-1.0, -1.0), v(1.0, -1.0), v(1.0, 1.0), v(-1.0, 1.0)],
            indices: vec![0, 1, 2, 0, 2, 3],
            material_slot: None,
            skin: Vec::new(),
        };
        let mesh = MeshAsset::new(vec![sm], vec![]);
        let id = proj
            .write_asset(&d, "Quad", &mesh, None, vec![], None)
            .unwrap();
        let proj = Arc::new(Mutex::new(proj));

        let mut thumb = Thumbnailer::new(64);
        if thumb.gpu().is_none() {
            eprintln!("no GPU adapter; skipping mesh thumbnail render");
            return;
        }
        let rgba = thumb.render_rgba(&proj, id).expect("mesh thumbnail");
        assert_eq!(rgba.len(), 64 * 64 * 4);
        // The quad must actually be drawn: some pixels brighter than the
        // studio-grey backdrop (~sRGB 96) — proves the camera frames geometry.
        let lit = rgba.chunks_exact(4).any(|p| p[0] > 130 && p[1] > 130);
        assert!(lit, "expected the lit quad to appear in the thumbnail");
    }
}
