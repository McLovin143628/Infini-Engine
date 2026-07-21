//! The renderer's input: a flat, engine-agnostic scene description.
//!
//! Phase 2 scope: unit-cube instances with f64 world transforms (ECS binding
//! arrives in Phase 3 — the host converts whatever it has into this). The
//! `version` counter gates GPU re-uploads: bump it on any instance change.

use glam::{DVec3, Quat, Vec3};

use crate::debug_draw::DebugDraw;

pub use inf_render_2d::{
    PrebatchedRun, RenderChunk, RenderTilemap, SpriteInstance, TextureHandle, TilemapParams,
};

/// A one-shot request to upload an RGBA8 texture into the sprite pass's GPU
/// cache, keyed by [`TextureHandle`]. The pass dedups by handle, so re-listing
/// an already-uploaded texture is a cheap no-op. Straight RGBA8 rows,
/// `width*height*4` bytes, sRGB-encoded base color.
#[derive(Debug, Clone, PartialEq)]
pub struct SpriteTextureUpload {
    pub handle: TextureHandle,
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

/// Reserved instance id meaning "nothing" (ID buffer clear value).
pub const ID_NONE: u32 = 0;

/// Instance ids at or above this are gizmo parts, not scene objects
/// (see `gizmo.rs`).
pub const ID_GIZMO_BASE: u32 = 0xffff_ff00;

#[derive(Debug, Clone, Copy)]
pub struct MeshInstance {
    /// World-space translation (f64 — architecture rule 3).
    pub translation: DVec3,
    pub rotation: Quat,
    pub scale: Vec3,
    /// Linear-space base color (rgba).
    pub color: [f32; 4],
    /// Metallic-roughness PBR parameters.
    pub metallic: f32,
    pub roughness: f32,
    /// Linear self-emitted color (rgb).
    pub emissive: [f32; 3],
    /// Stable pick id; `ID_NONE` is reserved, ids ≥ `ID_GIZMO_BASE` are
    /// reserved for gizmo parts.
    pub id: u32,
}

impl MeshInstance {
    /// A plain lit instance (metallic 0, roughness 0.5, no emission) — the
    /// common case for tests and simple callers.
    pub fn lit(translation: DVec3, rotation: Quat, scale: Vec3, color: [f32; 4], id: u32) -> Self {
        Self {
            translation,
            rotation,
            scale,
            color,
            metallic: 0.0,
            roughness: 0.5,
            emissive: [0.0; 3],
            id,
        }
    }
}

/// Directional vs point light (spot is projected as point for now).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightKind {
    Directional,
    Point,
}

/// A scene light in world space. The mesh pass converts point positions to
/// render-local (floating-origin-relative) space at upload, exactly like
/// instance transforms.
#[derive(Debug, Clone, Copy)]
pub struct RenderLight {
    pub kind: LightKind,
    /// Linear light color.
    pub color: [f32; 3],
    /// Radiant intensity multiplier.
    pub intensity: f32,
    /// Unit direction *toward* the light (directional only).
    pub direction: Vec3,
    /// World-space position (point only).
    pub position: DVec3,
    /// Influence radius in metres (point only); 0 ⇒ unbounded.
    pub range: f32,
}

/// A minimal 2D light (P8.1c): a soft radial falloff in the sprite plane. The
/// sprite pass converts `position` to render-local (floating-origin-relative) at
/// upload, exactly like 3D point lights, and lights every sprite/tile/text/
/// 9-slice fragment by world-XY distance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderLight2D {
    /// Linear light color.
    pub color: [f32; 3],
    /// Brightness multiplier.
    pub intensity: f32,
    /// World-space falloff radius; the contribution is `smoothstep(radius, 0,
    /// dist)`, so it is full at the light and zero at/after `radius`.
    pub radius: f32,
    /// World-space position (the sprite plane's XY is what matters).
    pub position: DVec3,
}

/// Scene-level 2D ambient term. **Defaults to white (`1,1,1`)** so that with no
/// [`RenderLight2D`] present every sprite renders exactly as before
/// (`texel·tint·1`) — the byte-stability guarantee for pre-P8.1c goldens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ambient2D(pub [f32; 3]);

impl Default for Ambient2D {
    fn default() -> Self {
        Self([1.0, 1.0, 1.0])
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkyParams {
    /// Linear colors.
    pub zenith: [f32; 3],
    pub horizon: [f32; 3],
    pub ground: [f32; 3],
}

impl Default for SkyParams {
    fn default() -> Self {
        // Editor-dark sky tuned to the infinity-dark theme.
        Self {
            zenith: [0.012, 0.021, 0.038],
            horizon: [0.055, 0.081, 0.120],
            ground: [0.009, 0.011, 0.015],
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct RenderScene {
    pub instances: Vec<MeshInstance>,
    /// Scene lights (directional + point). Empty ⇒ the shader falls back to a
    /// default editor sun so unlit demo scenes still render.
    pub lights: Vec<RenderLight>,
    /// 2D sprites (batched + drawn by the sprite pass over the 3D scene).
    pub sprites: Vec<SpriteInstance>,
    /// 2D tilemaps (P8.1b). The sprite pass culls each tilemap's chunks against
    /// the camera and expands the visible ones into prebatched sprite runs, then
    /// batches them together with the loose `sprites`. Because culling depends on
    /// the live camera, the pass re-expands tilemaps every frame (not gated by
    /// `version`) while any tilemap is present — a documented v1 cost (a
    /// camera-delta / dirty-region optimization is a follow-up).
    pub tilemaps: Vec<RenderTilemap>,
    /// Host-expanded 2D primitives that are already in draw order and share one
    /// `(layer, order, texture)` per run — 9-slices (nine quads) and text blocks
    /// (one quad per glyph), expanded by `inf-render-2d`. The sprite pass merges
    /// these with the loose `sprites` and the tilemap runs in one painter sort.
    /// Version-gated (the host rebuilds them on document change), unlike
    /// tilemaps which additionally re-expand per frame for culling.
    pub prebatched: Vec<PrebatchedRun>,
    /// Minimal 2D lights (P8.1c). Empty ⇒ only `ambient_2d` lights the sprites.
    pub lights_2d: Vec<RenderLight2D>,
    /// Scene-level 2D ambient (default white → unlit sprites unchanged).
    pub ambient_2d: Ambient2D,
    /// Textures to hand to the sprite pass's GPU cache (drained/deduped by
    /// handle). The host populates this once per newly-referenced texture.
    pub pending_texture_uploads: Vec<SpriteTextureUpload>,
    /// Bump on every change to `instances`/`lights`/`sprites`/`tilemaps` — gates
    /// buffer re-upload (tilemaps additionally re-expand per frame for culling).
    pub version: u64,
    pub sky: SkyParams,
    pub grid_enabled: bool,
    /// Ids drawn with the selection outline.
    pub selected: Vec<u32>,
    /// Id drawn with the hover outline (weaker), if any.
    pub hovered: Option<u32>,
    /// Immediate-mode debug lines, rebuilt by the host each frame
    /// (render-local space — not gated by `version`).
    pub debug: DebugDraw,
}

impl RenderScene {
    pub fn mark_dirty(&mut self) {
        self.version = self.version.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dirty_bumps_version() {
        let mut s = RenderScene::default();
        let v0 = s.version;
        s.instances.push(MeshInstance::lit(
            DVec3::ZERO,
            Quat::IDENTITY,
            Vec3::ONE,
            [1.0; 4],
            1,
        ));
        s.mark_dirty();
        assert_ne!(s.version, v0);
    }
}
