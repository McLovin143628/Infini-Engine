//! Core scene components (P3.1.3).
//!
//! Every editable component derives `Component + Reflect + serde`, is registered
//! in [`crate::registry`], and reflects `Component + Default` so the Details
//! panel can read/write/reset it generically. Computed, non-editable components
//! (`GlobalTransform`, `ComputedVisibility`) are plain components refreshed by
//! transform propagation.

use bevy_ecs::prelude::*;
use bevy_ecs::reflect::ReflectComponent;
use bevy_reflect::std_traits::ReflectDefault;
use bevy_reflect::Reflect;
use glam::{DAffine3, DQuat, DVec3};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::math::{Color, Vec2d, Vec3d};

/// Stable, save-surviving identity. Bevy `Entity` ids are reused across
/// spawn/despawn and never persisted; the `Guid` is what `.inf_lvl` stores and
/// what selection/undo reference across a reload. Not reflected — identity is
/// not an editable property.
#[derive(Component, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Guid(pub Uuid);

impl Guid {
    pub fn new(uuid: Uuid) -> Self {
        Self(uuid)
    }
}

/// The Outliner label (UE's "actor label"). Shown in the Outliner + Details
/// header and renamed through a dedicated command, so it is registered for
/// reflection but excluded from the generic property grid.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[reflect(Component, Default)]
pub struct Name(pub String);

impl Name {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for Name {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Local transform: translation (metres), rotation (**euler degrees**), scale.
///
/// Rotation is euler degrees (X pitch, Y yaw, Z roll) exactly as UE presents it
/// — a quaternion in a numeric grid is hostile UX. Math goes through
/// [`Transform::affine`], which composes a `DQuat` via `from_euler(YXZ, …)`.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct Transform {
    pub translation: Vec3d,
    pub rotation: Vec3d,
    pub scale: Vec3d,
}

impl Transform {
    pub const IDENTITY: Self = Self {
        translation: Vec3d::ZERO,
        rotation: Vec3d::ZERO,
        scale: Vec3d::ONE,
    };

    pub fn from_translation(t: DVec3) -> Self {
        Self {
            translation: t.into(),
            ..Self::IDENTITY
        }
    }

    /// Rotation as a quaternion (YXZ euler order: yaw, pitch, roll).
    pub fn quat(self) -> DQuat {
        let r = self.rotation.to_dvec3();
        DQuat::from_euler(
            glam::EulerRot::YXZ,
            r.y.to_radians(),
            r.x.to_radians(),
            r.z.to_radians(),
        )
    }

    /// Replace the rotation from a quaternion (extracts YXZ euler degrees).
    pub fn set_quat(&mut self, q: DQuat) {
        let (y, x, z) = q.to_euler(glam::EulerRot::YXZ);
        self.rotation = Vec3d::new(x.to_degrees(), y.to_degrees(), z.to_degrees());
    }

    /// Local TRS as an affine (correct under non-uniform scale + rotation).
    pub fn affine(self) -> DAffine3 {
        DAffine3::from_scale_rotation_translation(
            self.scale.to_dvec3(),
            self.quat(),
            self.translation.to_dvec3(),
        )
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// World-space transform, recomputed by [`crate::transform::propagate`].
/// Not editable / not reflected.
#[derive(Component, Clone, Copy, Debug)]
pub struct GlobalTransform(pub DAffine3);

impl Default for GlobalTransform {
    fn default() -> Self {
        Self(DAffine3::IDENTITY)
    }
}

impl GlobalTransform {
    pub fn translation(&self) -> DVec3 {
        self.0.translation
    }
}

/// Self visibility toggle (the Outliner eye).
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct Visibility {
    pub visible: bool,
}

impl Default for Visibility {
    fn default() -> Self {
        Self { visible: true }
    }
}

/// Effective visibility (self AND every ancestor), set by propagation.
/// Not editable / not reflected.
#[derive(Component, Clone, Copy, Debug)]
pub struct ComputedVisibility(pub bool);

impl Default for ComputedVisibility {
    fn default() -> Self {
        Self(true)
    }
}

/// Built-in primitive mesh kinds (a placeholder for `MeshRef`-to-asset in
/// Phase 4). Enough to author the primitive+lights scene the phase gate needs.
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Primitive {
    #[default]
    Cube,
    Sphere,
    Plane,
    Cylinder,
    Cone,
}

/// A renderable mesh reference. Phase 4 adds an asset-GUID variant.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Default)]
#[reflect(Component, Default)]
pub struct MeshRef {
    pub primitive: Primitive,
}

/// Surface appearance: the metallic-roughness PBR parameter block (Phase 7).
/// New fields carry `#[serde(default)]` so pre-P7 `.inf_lvl` files that only
/// stored `base_color` still deserialize.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct Material {
    pub base_color: Color,
    /// 0 = dielectric, 1 = metal.
    #[serde(default = "default_metallic")]
    pub metallic: f32,
    /// Perceptual roughness, 0 = mirror-smooth, 1 = fully rough.
    #[serde(default = "default_roughness")]
    pub roughness: f32,
    /// Self-emitted color (rgb; alpha ignored). Black = non-emissive.
    #[serde(default = "default_emissive")]
    pub emissive: Color,
}

fn default_metallic() -> f32 {
    0.0
}
fn default_roughness() -> f32 {
    0.5
}
fn default_emissive() -> Color {
    Color::new(0.0, 0.0, 0.0, 1.0)
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color: Color::new(0.8, 0.8, 0.8, 1.0),
            metallic: default_metallic(),
            roughness: default_roughness(),
            emissive: default_emissive(),
        }
    }
}

/// Normalized UV sub-rect into a sprite's texture/atlas. Default = the full
/// texture `(0,0)-(1,1)`. As a nested struct it isn't surfaced in the generic
/// Details grid (the reflection walker only descends value types); atlas rects
/// are authored by the sprite-sheet slicer (P8.2) and always serde-persisted.
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Default)]
pub struct AtlasRect {
    pub min: Vec2d,
    pub max: Vec2d,
}

impl Default for AtlasRect {
    fn default() -> Self {
        Self {
            min: Vec2d::ZERO,
            max: Vec2d::ONE,
        }
    }
}

/// A 2D sprite: a textured, tinted, sortable quad. Rendered by the 2D pass
/// (`inf-render-2d` batcher + `inf-render` sprite pass) over the 3D scene.
///
/// Visibility follows the shared [`Visibility`]/`ComputedVisibility` components
/// (like meshes) — there is no per-sprite `visible` field.
///
/// Additive component (P8.1a): every new field carries `#[serde(default)]` so
/// levels saved before this component existed still load.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct Sprite {
    /// Texture/atlas asset GUID; `None` → the renderer's 1×1 white fallback.
    ///
    /// `#[reflect(ignore)]`: the reflection Details grid has no asset-ref widget
    /// yet (a follow-up shared with material texture refs), so the texture is
    /// assigned via drag-drop rather than typed. It is still serde-persisted.
    #[serde(default)]
    #[reflect(ignore)]
    pub texture: Option<Uuid>,
    /// Quad extent in world units (width, height).
    pub size: Vec2d,
    /// Normalized anchor in `[0,1]²`; `(0.5, 0.5)` centers the quad.
    pub pivot: Vec2d,
    /// Linear tint multiplied with the sampled texel (straight alpha).
    pub color: Color,
    /// Atlas UV sub-rect (defaults to the full texture).
    #[serde(default)]
    pub atlas_rect: AtlasRect,
    /// Coarse draw bucket (lower draws further back).
    #[serde(default)]
    pub sorting_layer: i32,
    /// Fine ordering within a layer (lower draws further back).
    #[serde(default)]
    pub order: i32,
    #[serde(default)]
    pub flip_x: bool,
    #[serde(default)]
    pub flip_y: bool,
}

impl Default for Sprite {
    fn default() -> Self {
        Self {
            texture: None,
            size: Vec2d::ONE,
            pivot: Vec2d::splat(0.5),
            color: Color::WHITE,
            atlas_rect: AtlasRect::default(),
            sorting_layer: 0,
            order: 0,
            flip_x: false,
            flip_y: false,
        }
    }
}

#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LightKind {
    #[default]
    Directional,
    Point,
    Spot,
}

/// A light source.
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct Light {
    pub kind: LightKind,
    pub color: Color,
    pub intensity: f32,
}

impl Default for Light {
    fn default() -> Self {
        Self {
            kind: LightKind::Point,
            color: Color::WHITE,
            intensity: 1.0,
        }
    }
}

/// A scene camera (distinct from the editor's fly camera).
#[derive(Component, Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[reflect(Component, Default)]
pub struct Camera {
    pub fov_y_deg: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            fov_y_deg: 60.0,
            near: 0.05,
            far: 10_000.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sprite_serde_round_trips() {
        let s = Sprite {
            texture: Some(Uuid::from_u128(0x1234_5678_9abc_def0_1122_3344_5566_7788)),
            size: Vec2d::new(2.0, 3.0),
            pivot: Vec2d::new(0.25, 0.75),
            color: Color::new(0.2, 0.4, 0.6, 0.8),
            atlas_rect: AtlasRect {
                min: Vec2d::new(0.1, 0.2),
                max: Vec2d::new(0.9, 0.8),
            },
            sorting_layer: -3,
            order: 5,
            flip_x: true,
            flip_y: false,
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: Sprite = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn sprite_defaults_fill_missing_fields() {
        // A pre-P8 payload predates every additive field: only the always-present
        // fields are stored. `#[serde(default)]` must reconstruct the rest.
        let minimal = r#"{
            "texture": null,
            "size": { "x": 1.0, "y": 1.0 },
            "pivot": { "x": 0.5, "y": 0.5 },
            "color": { "r": 1.0, "g": 1.0, "b": 1.0, "a": 1.0 }
        }"#;
        let s: Sprite = serde_json::from_str(minimal).unwrap();
        assert_eq!(s, Sprite::default());
        assert_eq!(s.atlas_rect, AtlasRect::default());
        assert_eq!(s.sorting_layer, 0);
    }
}
