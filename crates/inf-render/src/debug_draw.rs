//! Debug-primitive layer: immediate-mode lines rebuilt by the host each frame
//! (wireframe boxes, bounds, axes). Vertices are render-local f32 — the host
//! converts world positions through its floating origin while building.

use glam::{Quat, Vec2, Vec3};

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

/// A 2D collider shape for debug outlining (P8.3b). Mirrors `inf-ecs`'s
/// `ColliderShape2DKind` **without** depending on it — the viewport host maps a
/// `Collider2D` component to one of these and draws the outline into the
/// [`DebugDraw`] line layer for selected entities. Extents/radii are in the
/// collider's local XY frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColliderOutline2D {
    /// Axis-aligned box given by half-extents.
    Box { half: Vec2 },
    /// Circle of the given radius.
    Circle { radius: f32 },
    /// Vertical capsule: a segment of length `2*half_height` along local Y swept
    /// by `radius`.
    Capsule { half_height: f32, radius: f32 },
}

/// The closed-loop outline points of a 2D collider, in its local XY frame.
///
/// Connect consecutive points — and the last back to the first — to stroke the
/// outline. `circle_segments` is the segment count approximating a **full**
/// circle; each of a capsule's two semicircular caps uses `circle_segments / 2`
/// segments. Returns:
/// * **Box** → 4 corner points.
/// * **Circle** → `circle_segments` points around the rim.
/// * **Capsule** → `2 * (circle_segments/2 + 1)` points (two caps; the straight
///   sides emerge from connecting the cap endpoints).
pub fn collider_outline_2d(shape: ColliderOutline2D, circle_segments: u32) -> Vec<Vec2> {
    match shape {
        ColliderOutline2D::Box { half } => vec![
            Vec2::new(-half.x, -half.y),
            Vec2::new(half.x, -half.y),
            Vec2::new(half.x, half.y),
            Vec2::new(-half.x, half.y),
        ],
        ColliderOutline2D::Circle { radius } => {
            let n = circle_segments.max(3);
            (0..n)
                .map(|i| {
                    let a = std::f32::consts::TAU * (i as f32) / (n as f32);
                    Vec2::new(radius * a.cos(), radius * a.sin())
                })
                .collect()
        }
        ColliderOutline2D::Capsule {
            half_height,
            radius,
        } => {
            let cap = (circle_segments / 2).max(2);
            let mut pts = Vec::with_capacity(2 * (cap as usize + 1));
            // Top cap: angle 0 → π about centre (0, +half_height).
            for i in 0..=cap {
                let a = std::f32::consts::PI * (i as f32) / (cap as f32);
                pts.push(Vec2::new(radius * a.cos(), half_height + radius * a.sin()));
            }
            // Bottom cap: angle π → 2π about centre (0, -half_height).
            for i in 0..=cap {
                let a = std::f32::consts::PI * (1.0 + (i as f32) / (cap as f32));
                pts.push(Vec2::new(radius * a.cos(), -half_height + radius * a.sin()));
            }
            pts
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_outline_has_four_corners() {
        let pts = collider_outline_2d(
            ColliderOutline2D::Box {
                half: Vec2::new(2.0, 1.0),
            },
            32,
        );
        assert_eq!(pts.len(), 4);
        assert!(pts.contains(&Vec2::new(-2.0, -1.0)));
        assert!(pts.contains(&Vec2::new(2.0, 1.0)));
    }

    #[test]
    fn circle_outline_has_segment_count_points_on_the_rim() {
        let pts = collider_outline_2d(ColliderOutline2D::Circle { radius: 3.0 }, 32);
        assert_eq!(pts.len(), 32);
        // Every point lies on the circle of radius 3.
        for p in &pts {
            assert!((p.length() - 3.0).abs() < 1e-4, "point {p:?} off the rim");
        }
    }

    #[test]
    fn capsule_outline_caps_are_offset_by_half_height() {
        let h = 2.0;
        let r = 0.5;
        let pts = collider_outline_2d(
            ColliderOutline2D::Capsule {
                half_height: h,
                radius: r,
            },
            32,
        );
        // 2 * (32/2 + 1) = 34 points.
        assert_eq!(pts.len(), 34);
        // The topmost point sits at the top cap's apex (0, h + r).
        let top = pts
            .iter()
            .copied()
            .reduce(|a, b| if a.y > b.y { a } else { b })
            .unwrap();
        assert!(
            (top.x).abs() < 1e-4 && (top.y - (h + r)).abs() < 1e-4,
            "top {top:?}"
        );
        // The bottommost at the bottom cap's apex (0, -h - r).
        let bot = pts
            .iter()
            .copied()
            .reduce(|a, b| if a.y < b.y { a } else { b })
            .unwrap();
        assert!(
            (bot.x).abs() < 1e-4 && (bot.y - (-h - r)).abs() < 1e-4,
            "bot {bot:?}"
        );
        // First and last points are the right-side cap endpoints (± half_height).
        assert!((pts[0] - Vec2::new(r, h)).length() < 1e-4);
        assert!((pts[pts.len() - 1] - Vec2::new(r, -h)).length() < 1e-4);
    }

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
