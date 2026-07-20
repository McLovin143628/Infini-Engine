//! Debug-primitive layer: immediate-mode lines rebuilt by the host each frame
//! (wireframe boxes, bounds, axes). Vertices are render-local f32 — the host
//! converts world positions through its floating origin while building.

use glam::{Quat, Vec3};

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DebugVertex {
    pub pos: [f32; 3],
    pub color: [f32; 4],
}

#[derive(Debug, Clone, Default)]
pub struct DebugDraw {
    pub verts: Vec<DebugVertex>,
}

impl DebugDraw {
    pub fn clear(&mut self) {
        self.verts.clear();
    }

    pub fn line(&mut self, a: Vec3, b: Vec3, color: [f32; 4]) {
        self.verts.push(DebugVertex {
            pos: a.to_array(),
            color,
        });
        self.verts.push(DebugVertex {
            pos: b.to_array(),
            color,
        });
    }

    /// Oriented wireframe box (12 edges).
    pub fn wire_box(&mut self, center: Vec3, half: Vec3, rot: Quat, color: [f32; 4]) {
        let corner = |sx: f32, sy: f32, sz: f32| center + rot * (half * Vec3::new(sx, sy, sz));
        let c = [
            corner(-1.0, -1.0, -1.0),
            corner(1.0, -1.0, -1.0),
            corner(1.0, -1.0, 1.0),
            corner(-1.0, -1.0, 1.0),
            corner(-1.0, 1.0, -1.0),
            corner(1.0, 1.0, -1.0),
            corner(1.0, 1.0, 1.0),
            corner(-1.0, 1.0, 1.0),
        ];
        const EDGES: [(usize, usize); 12] = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ];
        for (a, b) in EDGES {
            self.line(c[a], c[b], color);
        }
    }

    /// RGB axis tripod at `origin`.
    pub fn axes(&mut self, origin: Vec3, len: f32) {
        self.line(origin, origin + Vec3::X * len, [0.95, 0.28, 0.30, 1.0]);
        self.line(origin, origin + Vec3::Y * len, [0.45, 0.85, 0.30, 1.0]);
        self.line(origin, origin + Vec3::Z * len, [0.25, 0.45, 1.0, 1.0]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_box_emits_12_edges() {
        let mut d = DebugDraw::default();
        d.wire_box(Vec3::ZERO, Vec3::ONE, Quat::IDENTITY, [1.0; 4]);
        assert_eq!(d.verts.len(), 24);
    }

    #[test]
    fn vertex_layout_is_tightly_packed() {
        assert_eq!(std::mem::size_of::<DebugVertex>(), 28);
    }
}
