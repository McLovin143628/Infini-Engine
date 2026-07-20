//! The renderer's input: a flat, engine-agnostic scene description.
//!
//! Phase 2 scope: unit-cube instances with f64 world transforms (ECS binding
//! arrives in Phase 3 — the host converts whatever it has into this). The
//! `version` counter gates GPU re-uploads: bump it on any instance change.

use glam::{DVec3, Quat, Vec3};

use crate::debug_draw::DebugDraw;

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
    /// Linear-space base color.
    pub color: [f32; 4],
    /// Stable pick id; `ID_NONE` is reserved, ids ≥ `ID_GIZMO_BASE` are
    /// reserved for gizmo parts.
    pub id: u32,
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
    /// Bump on every change to `instances` — gates instance-buffer re-upload.
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
        s.instances.push(MeshInstance {
            translation: DVec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            color: [1.0; 4],
            id: 1,
        });
        s.mark_dirty();
        assert_ne!(s.version, v0);
    }
}
