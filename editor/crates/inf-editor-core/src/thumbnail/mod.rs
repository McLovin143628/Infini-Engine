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

/// Renders asset thumbnails. Holds a lazily-created, possibly-absent GPU context.
pub struct Thumbnailer {
    size: u32,
    gpu: GpuState,
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
        }
    }

    /// Lazily obtain the GPU context; `None` if this machine has no adapter.
    fn gpu(&mut self) -> Option<&GpuContext> {
        if matches!(self.gpu, GpuState::Uninit) {
            self.gpu = match GpuContext::headless() {
                Ok(ctx) => GpuState::Ready(Box::new(ctx)),
                Err(e) => {
                    tracing::info!("thumbnailer: no GPU adapter ({e}); mesh/material previews off");
                    GpuState::Unavailable
                }
            };
        }
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
                let tex: TextureAsset = load_payload(&path).ok()?;
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
    pub fn render_material_preview(
        &mut self,
        surface_wgsl: &str,
        tex_count: u32,
        size: u32,
    ) -> Option<Vec<u8>> {
        let gpu = self.gpu()?;
        scene_render::render_material_preview(gpu, size, surface_wgsl, tex_count).ok()
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
pub fn encode_png(size: u32, rgba: &[u8]) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut buf, size, size);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
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
