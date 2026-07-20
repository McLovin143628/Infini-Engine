//! Reflected f64 value types for scene components.
//!
//! We deliberately do **not** enable `bevy_reflect`'s `glam` feature: it pins
//! glam 0.32, and the renderer pins 0.33 — enabling it would put two glam
//! versions in the tree. Instead the editable transform is stored in these
//! plain f64 value types (which derive `Reflect` + `serde`) and converted to
//! glam only for math. Architecture rule 3 (f64 world / f32 render) is
//! preserved: translations are f64 all the way to the floating-origin split.

use bevy_reflect::Reflect;
use glam::DVec3;
use serde::{Deserialize, Serialize};

/// A 3-component f64 vector. Editable as a `vec3` widget in the Details panel.
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct Vec3d {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3d {
    pub const ZERO: Self = Self::splat(0.0);
    pub const ONE: Self = Self::splat(1.0);

    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub const fn splat(v: f64) -> Self {
        Self { x: v, y: v, z: v }
    }

    pub fn to_dvec3(self) -> DVec3 {
        DVec3::new(self.x, self.y, self.z)
    }

    pub fn from_dvec3(v: DVec3) -> Self {
        Self::new(v.x, v.y, v.z)
    }
}

impl Default for Vec3d {
    fn default() -> Self {
        Self::ZERO
    }
}

impl From<DVec3> for Vec3d {
    fn from(v: DVec3) -> Self {
        Self::from_dvec3(v)
    }
}

impl From<Vec3d> for DVec3 {
    fn from(v: Vec3d) -> Self {
        v.to_dvec3()
    }
}

/// Linear RGBA colour, editable as a `color` widget.
#[derive(Reflect, Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const WHITE: Self = Self::new(1.0, 1.0, 1.0, 1.0);

    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub fn to_array(self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }
}

impl Default for Color {
    fn default() -> Self {
        Self::WHITE
    }
}
