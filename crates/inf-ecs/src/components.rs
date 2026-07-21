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

use crate::math::{Color, Vec3d};

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
