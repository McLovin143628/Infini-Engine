//! Transform gizmos: engine-rendered translate/rotate/scale handles with
//! constant screen size, axis/plane constraints, and snapping.
//!
//! Split cleanly into (a) pure interaction math — screen ray → constrained
//! world delta, unit-tested headless — and (b) geometry generation as debug
//! lines. Hit-testing is analytic (screen-space distance to each handle)
//! rather than an ID-buffer pass: thin handles pick far more reliably that way,
//! and it keeps picking on the CPU where the drag math already lives.

use glam::{DVec3, Mat4, Quat, Vec2, Vec3, Vec4Swizzles};

use crate::debug_draw::DebugDraw;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoMode {
    Translate,
    Rotate,
    Scale,
}

/// Which handle the pointer is over / dragging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoAxis {
    X,
    Y,
    Z,
    /// Screen-facing plane handles (translate/scale): drag in the plane whose
    /// normal is the named axis.
    PlaneX,
    PlaneY,
    PlaneZ,
}

impl GizmoAxis {
    /// Unit direction of the primary axis (planes report their normal).
    pub fn dir(self) -> Vec3 {
        match self {
            GizmoAxis::X | GizmoAxis::PlaneX => Vec3::X,
            GizmoAxis::Y | GizmoAxis::PlaneY => Vec3::Y,
            GizmoAxis::Z | GizmoAxis::PlaneZ => Vec3::Z,
        }
    }

    pub fn is_plane(self) -> bool {
        matches!(
            self,
            GizmoAxis::PlaneX | GizmoAxis::PlaneY | GizmoAxis::PlaneZ
        )
    }

    pub fn color(self) -> [f32; 4] {
        match self {
            GizmoAxis::X | GizmoAxis::PlaneX => [0.95, 0.26, 0.28, 1.0],
            GizmoAxis::Y | GizmoAxis::PlaneY => [0.45, 0.85, 0.30, 1.0],
            GizmoAxis::Z | GizmoAxis::PlaneZ => [0.25, 0.48, 1.0, 1.0],
        }
    }
}

/// Screen-constant world size of the gizmo, given the pivot's distance from
/// the eye and the vertical fov: keeps the handles ~`GIZMO_SCREEN_FRAC` of the
/// viewport tall regardless of zoom.
pub const GIZMO_SCREEN_FRAC: f32 = 0.16;

pub fn gizmo_world_size(pivot: Vec3, eye: Vec3, fov_y: f32) -> f32 {
    let dist = (pivot - eye).length().max(0.01);
    // Half-height of the view frustum at `dist`, times the target fraction.
    dist * (fov_y * 0.5).tan() * 2.0 * GIZMO_SCREEN_FRAC
}

/// Screen-constant gizmo size for the orthographic (2D) camera: the world size
/// that fills `GIZMO_SCREEN_FRAC` of the viewport height. In ortho the on-screen
/// size is independent of distance, so it depends only on the view half-height
/// (the zoom).
pub fn gizmo_world_size_ortho(half_height: f32) -> f32 {
    half_height * 2.0 * GIZMO_SCREEN_FRAC
}

/// A drag in progress: the handle and the world-space anchor where it started.
#[derive(Debug, Clone, Copy)]
pub struct GizmoDrag {
    pub mode: GizmoMode,
    pub axis: GizmoAxis,
    /// Orientation basis for the gizmo axes: `IDENTITY` = world-aligned, or the
    /// selection's world rotation for a local-space gizmo. Captured at drag
    /// start and held fixed for the whole gesture so a rotate drag doesn't chase
    /// its own basis. Every consumer of [`GizmoAxis::dir`] applies `basis * dir`.
    pub basis: Quat,
    /// Pivot (selection center) at drag start, render-local.
    pub origin: Vec3,
    /// The point on the constraint line/plane under the cursor at drag start.
    pub grab: Vec3,
    /// Reference vector for rotation (origin→grab projected into the plane).
    pub rot_ref: Vec3,
}

/// Result of one drag update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GizmoDelta {
    /// World-space translation to add to the selection.
    Translate(DVec3),
    /// Rotation about `axis` (render-local, through the pivot), radians.
    Rotate { axis: Vec3, radians: f32 },
    /// Per-axis scale multiplier (1.0 = unchanged).
    Scale(Vec3),
}

/// Closest point on the infinite line `p + t·d` to the ray `o + s·rd`.
///
/// # `denom.abs() < 1e-6` is false for NaN (C4-44)
///
/// Both this and [`ray_plane`] guarded a division with a magnitude test, which
/// is the house's standing NaN-blind-guard class: `NaN < 1e-6` is false, so the
/// "degenerate" branch was skipped and the division ran. The result flows
/// `GizmoDrag::update` → `GizmoDelta::Translate` → `EngineHost::apply_delta` →
/// `SceneDoc` → **`.inf_lvl`**, with no finiteness check anywhere on the chain.
/// Rotate and Scale happen to be safe, which is what left Translate exposed.
///
/// Asked the other way round — "is this a usable divisor" — NaN answers
/// correctly and takes the degenerate branch it belongs in.
fn closest_on_line(p: Vec3, d: Vec3, o: Vec3, rd: Vec3) -> Vec3 {
    // Standard closest-points-of-two-lines solution.
    let w0 = p - o;
    let a = d.dot(d);
    let b = d.dot(rd);
    let c = rd.dot(rd);
    let dd = d.dot(w0);
    let e = rd.dot(w0);
    let denom = a * c - b * b;
    let t = if denom.is_finite() && denom.abs() >= 1e-6 {
        (b * e - c * dd) / denom
    } else {
        0.0
    };
    let out = p + d * t;
    if out.is_finite() {
        out
    } else {
        p
    }
}

/// Intersect ray `o + s·rd` with the plane through `p` with normal `n`.
///
/// Returns `None` for a degenerate or non-finite intersection — see
/// [`closest_on_line`] for why the guard is written as a positive test. `s <
/// 0.0` is NaN-blind in the same way, so this used to hand back `Some(NaN)` as
/// an "intersection".
fn ray_plane(o: Vec3, rd: Vec3, p: Vec3, n: Vec3) -> Option<Vec3> {
    let denom = n.dot(rd);
    if !(denom.is_finite() && denom.abs() >= 1e-6) {
        return None;
    }
    let s = n.dot(p - o) / denom;
    if !(s >= 0.0) {
        return None;
    }
    let hit = o + rd * s;
    hit.is_finite().then_some(hit)
}

/// Where the cursor ray grabs a handle at drag start. `basis` rotates the axis
/// directions (world-aligned when `IDENTITY`, the selection's rotation in local
/// mode).
pub fn grab_point(
    mode: GizmoMode,
    axis: GizmoAxis,
    basis: Quat,
    origin: Vec3,
    ro: Vec3,
    rd: Vec3,
) -> Vec3 {
    let dir = basis * axis.dir();
    match (mode, axis.is_plane()) {
        (GizmoMode::Rotate, _) => ray_plane(ro, rd, origin, dir).unwrap_or(origin + dir),
        (_, true) => ray_plane(ro, rd, origin, dir).unwrap_or(origin),
        (_, false) => closest_on_line(origin, dir, ro, rd),
    }
}

impl GizmoDrag {
    pub fn begin(
        mode: GizmoMode,
        axis: GizmoAxis,
        basis: Quat,
        origin: Vec3,
        ro: Vec3,
        rd: Vec3,
    ) -> Self {
        let grab = grab_point(mode, axis, basis, origin, ro, rd);
        let rot_ref = (grab - origin).normalize_or_zero();
        Self {
            mode,
            axis,
            basis,
            origin,
            grab,
            rot_ref,
        }
    }

    /// Update from the current cursor ray; `snap` (>0) quantizes the result
    /// (metres for translate, radians for rotate, ratio for scale).
    pub fn update(&self, ro: Vec3, rd: Vec3, snap: f32) -> GizmoDelta {
        match self.mode {
            GizmoMode::Translate => {
                let now = grab_point(self.mode, self.axis, self.basis, self.origin, ro, rd);
                let mut delta = now - self.grab;
                if !self.axis.is_plane() {
                    // Constrain to the single (basis-rotated) axis.
                    let d = self.basis * self.axis.dir();
                    delta = d * delta.dot(d);
                }
                if snap > 0.0 {
                    delta = (delta / snap).round() * snap;
                }
                GizmoDelta::Translate(delta.as_dvec3())
            }
            GizmoMode::Rotate => {
                let n = self.basis * self.axis.dir();
                let now = ray_plane(ro, rd, self.origin, n).unwrap_or(self.origin + self.rot_ref);
                let cur = (now - self.origin).normalize_or_zero();
                let mut ang = self.rot_ref.cross(cur).dot(n).atan2(self.rot_ref.dot(cur));
                if snap > 0.0 {
                    ang = (ang / snap).round() * snap;
                }
                GizmoDelta::Rotate {
                    axis: n,
                    radians: ang,
                }
            }
            GizmoMode::Scale => {
                let now = grab_point(
                    GizmoMode::Translate,
                    self.axis,
                    self.basis,
                    self.origin,
                    ro,
                    rd,
                );
                let start_len = (self.grab - self.origin).length().max(1e-4);
                let now_len = (now - self.origin).length();
                let mut ratio = now_len / start_len;
                if snap > 0.0 {
                    ratio = (ratio / snap).round() * snap;
                }
                ratio = ratio.max(0.01);
                // Scale is applied component-wise in the object's LOCAL frame, so
                // the factor uses the un-rotated axis; only the drag measurement
                // (above) follows the basis-rotated handle direction.
                let factor = if self.axis.is_plane() {
                    Vec3::splat(ratio) // plane handle = uniform scale
                } else {
                    Vec3::ONE + self.axis.dir() * (ratio - 1.0)
                };
                GizmoDelta::Scale(factor)
            }
        }
    }
}

/// Project a render-local point to pixel coordinates; `None` if behind the eye.
fn project(p: Vec3, view_proj: Mat4, width: f32, height: f32) -> Option<Vec2> {
    let clip = view_proj * p.extend(1.0);
    if clip.w <= 1e-6 {
        return None;
    }
    let ndc = clip.xy() / clip.w;
    Some(Vec2::new(
        (ndc.x * 0.5 + 0.5) * width,
        (0.5 - ndc.y * 0.5) * height,
    ))
}

/// Analytic hit-test: the handle whose screen projection is closest to the
/// cursor within `PICK_PIXELS`, or `None`. `axes` are the handles to test in
/// priority order (planes first — they sit near the center).
///
/// `two_d` restricts the handle set to the 2D editor gizmo: translate/scale
/// expose X, Y and the screen-facing XY plane handle (`PlaneZ`); rotate exposes
/// only the Z ring — matching [`build_geometry`].
#[allow(clippy::too_many_arguments)]
pub fn pick_axis(
    mode: GizmoMode,
    origin: Vec3,
    basis: Quat,
    size: f32,
    view_proj: Mat4,
    cursor: Vec2,
    width: f32,
    height: f32,
    two_d: bool,
) -> Option<GizmoAxis> {
    const PICK_PIXELS: f32 = 11.0;
    let o = project(origin, view_proj, width, height)?;
    // Basis-rotated tangents for a plane whose local normal is `axis.dir()`.
    let tangents = |axis: GizmoAxis| {
        let (u, v) = plane_tangents(axis.dir());
        (basis * u, basis * v)
    };

    let mut best: Option<(GizmoAxis, f32)> = None;
    let mut consider = |axis: GizmoAxis, dist: f32| {
        if dist <= PICK_PIXELS && best.is_none_or(|(_, b)| dist < b) {
            best = Some((axis, dist));
        }
    };

    let axes: &[GizmoAxis] = if two_d {
        &[GizmoAxis::X, GizmoAxis::Y]
    } else {
        &[GizmoAxis::X, GizmoAxis::Y, GizmoAxis::Z]
    };

    if mode == GizmoMode::Rotate {
        // Rotate: distance to each axis circle (radius = size) in screen space.
        // 2D constrains to the Z ring (rotation in the XY plane).
        let rings: &[GizmoAxis] = if two_d {
            &[GizmoAxis::Z]
        } else {
            &[GizmoAxis::X, GizmoAxis::Y, GizmoAxis::Z]
        };
        for &a in rings {
            let (u, v) = tangents(a);
            let d = circle_screen_distance(origin, u, v, size, view_proj, cursor, width, height);
            if let Some(d) = d {
                consider(a, d);
            }
        }
        return best.map(|(a, _)| a);
    }

    // Translate/scale: plane handles (small quad near center) then axis lines.
    let planes: &[GizmoAxis] = if two_d {
        &[GizmoAxis::PlaneZ]
    } else {
        &[GizmoAxis::PlaneX, GizmoAxis::PlaneY, GizmoAxis::PlaneZ]
    };
    for &axis in planes {
        let (u, v) = tangents(axis);
        let corner = origin + (u + v) * size * 0.35;
        if let Some(c) = project(corner, view_proj, width, height) {
            consider(axis, cursor.distance(c));
        }
    }
    for &a in axes {
        let tip = project(origin + (basis * a.dir()) * size, view_proj, width, height);
        if let Some(tip) = tip {
            consider(a, point_segment_distance(cursor, o, tip));
        }
    }
    best.map(|(a, _)| a)
}

fn plane_tangents(normal: Vec3) -> (Vec3, Vec3) {
    match normal {
        v if v == Vec3::X => (Vec3::Y, Vec3::Z),
        v if v == Vec3::Y => (Vec3::X, Vec3::Z),
        _ => (Vec3::X, Vec3::Y),
    }
}

fn point_segment_distance(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let len2 = ab.length_squared();
    if len2 < 1e-6 {
        return p.distance(a);
    }
    let t = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

#[allow(clippy::too_many_arguments)]
fn circle_screen_distance(
    center: Vec3,
    u: Vec3,
    v: Vec3,
    radius: f32,
    view_proj: Mat4,
    cursor: Vec2,
    width: f32,
    height: f32,
) -> Option<f32> {
    let mut best = f32::MAX;
    let mut any = false;
    let segments = 48;
    let mut prev: Option<Vec2> = None;
    for i in 0..=segments {
        let a = i as f32 / segments as f32 * std::f32::consts::TAU;
        let p = center + (u * a.cos() + v * a.sin()) * radius;
        if let Some(sp) = project(p, view_proj, width, height) {
            if let Some(pp) = prev {
                best = best.min(point_segment_distance(cursor, pp, sp));
                any = true;
            }
            prev = Some(sp);
        } else {
            prev = None;
        }
    }
    any.then_some(best)
}

/// Emit the gizmo geometry as debug lines at `origin` (render-local),
/// highlighting `active` (hovered/dragged) if any.
pub fn build_geometry(
    draw: &mut DebugDraw,
    mode: GizmoMode,
    origin: Vec3,
    basis: Quat,
    size: f32,
    active: Option<GizmoAxis>,
    two_d: bool,
) {
    let hi = |axis: GizmoAxis, base: [f32; 4]| -> [f32; 4] {
        if active == Some(axis) {
            [1.0, 0.85, 0.15, 1.0] // amber highlight
        } else {
            base
        }
    };

    if two_d {
        // The 2D editor gizmo is always world-aligned (basis forced to identity
        // by the host in 2D mode).
        build_geometry_2d(draw, mode, origin, size, &hi);
        return;
    }

    match mode {
        GizmoMode::Translate | GizmoMode::Scale => {
            for (axis, plane_axis) in [
                (GizmoAxis::X, GizmoAxis::PlaneX),
                (GizmoAxis::Y, GizmoAxis::PlaneY),
                (GizmoAxis::Z, GizmoAxis::PlaneZ),
            ] {
                let d = basis * axis.dir();
                let (u, v) = plane_tangents(axis.dir());
                let (u, v) = (basis * u, basis * v);
                let color = hi(axis, axis.color());
                draw.line(origin, origin + d * size, color);
                if mode == GizmoMode::Scale {
                    // Small box at the tip.
                    draw.wire_box(origin + d * size, Vec3::splat(size * 0.06), basis, color);
                } else {
                    // Arrowhead as a little cross near the tip.
                    let tip = origin + d * size;
                    let back = tip - d * size * 0.12;
                    draw.line(tip, back + u * size * 0.05, color);
                    draw.line(tip, back - u * size * 0.05, color);
                    draw.line(tip, back + v * size * 0.05, color);
                    draw.line(tip, back - v * size * 0.05, color);
                }
                // Plane handle: an L near the origin in this axis's plane.
                let pc = hi(plane_axis, plane_axis.color());
                let a = origin + u * size * 0.35;
                let b = origin + v * size * 0.35;
                let corner = origin + (u + v) * size * 0.35;
                draw.line(a, corner, pc);
                draw.line(b, corner, pc);
            }
        }
        GizmoMode::Rotate => {
            for axis in [GizmoAxis::X, GizmoAxis::Y, GizmoAxis::Z] {
                let (u, v) = plane_tangents(axis.dir());
                let (u, v) = (basis * u, basis * v);
                let color = hi(axis, axis.color());
                let segments = 64;
                let mut prev = origin + u * size;
                for i in 1..=segments {
                    let a = i as f32 / segments as f32 * std::f32::consts::TAU;
                    let p = origin + (u * a.cos() + v * a.sin()) * size;
                    draw.line(prev, p, color);
                    prev = p;
                }
            }
        }
    }
}

/// 2D-mode gizmo geometry: translate/scale expose only the X and Y axes plus a
/// single screen-facing XY plane handle (`PlaneZ`); rotate is the Z ring alone.
/// The Z axis is intentionally absent (out of the sprite plane).
fn build_geometry_2d(
    draw: &mut DebugDraw,
    mode: GizmoMode,
    origin: Vec3,
    size: f32,
    hi: &impl Fn(GizmoAxis, [f32; 4]) -> [f32; 4],
) {
    match mode {
        GizmoMode::Translate | GizmoMode::Scale => {
            for axis in [GizmoAxis::X, GizmoAxis::Y] {
                let d = axis.dir();
                let color = hi(axis, axis.color());
                draw.line(origin, origin + d * size, color);
                if mode == GizmoMode::Scale {
                    draw.wire_box(
                        origin + d * size,
                        Vec3::splat(size * 0.06),
                        glam::Quat::IDENTITY,
                        color,
                    );
                } else {
                    let (u, v) = plane_tangents(d);
                    let tip = origin + d * size;
                    let back = tip - d * size * 0.12;
                    draw.line(tip, back + u * size * 0.05, color);
                    draw.line(tip, back - u * size * 0.05, color);
                    draw.line(tip, back + v * size * 0.05, color);
                    draw.line(tip, back - v * size * 0.05, color);
                }
            }
            // XY plane handle (PlaneZ): an L in the XY plane near the origin —
            // free move (translate) / uniform scale.
            let pc = hi(GizmoAxis::PlaneZ, GizmoAxis::PlaneZ.color());
            let (u, v) = (Vec3::X, Vec3::Y);
            let a = origin + u * size * 0.35;
            let b = origin + v * size * 0.35;
            let corner = origin + (u + v) * size * 0.35;
            draw.line(a, corner, pc);
            draw.line(b, corner, pc);
        }
        GizmoMode::Rotate => {
            let axis = GizmoAxis::Z;
            let (u, v) = plane_tangents(axis.dir());
            let color = hi(axis, axis.color());
            let segments = 64;
            let mut prev = origin + u * size;
            for i in 1..=segments {
                let a = i as f32 / segments as f32 * std::f32::consts::TAU;
                let p = origin + (u * a.cos() + v * a.sin()) * size;
                draw.line(prev, p, color);
                prev = p;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **C4-44 — the gizmo's two divisions were guarded NaN-blind**, and the
    /// value they produce is written into the `.inf_lvl`.
    ///
    /// `denom.abs() < 1e-6` is false for NaN, so the degenerate branch was
    /// skipped and the division ran; `ray_plane`'s `s < 0.0` is false for NaN
    /// too, so it returned `Some(NaN)` as an "intersection". From there:
    /// `GizmoDrag::update` → `GizmoDelta::Translate` → `EngineHost::apply_delta`
    /// → `SceneDoc` → the level file, unchecked the whole way.
    ///
    /// Un-fix mutation: restore either `< 1e-6` guard and the matching assertion
    /// below fails.
    #[test]
    fn a_degenerate_or_non_finite_ray_never_produces_a_nan_position() {
        let nan = Vec3::splat(f32::NAN);
        // A degenerate axis and a NaN ray both take the safe branch.
        let p = closest_on_line(Vec3::ZERO, Vec3::ZERO, Vec3::Z * -5.0, Vec3::Z);
        assert!(p.is_finite(), "degenerate axis produced {p:?}");
        let p = closest_on_line(Vec3::ZERO, Vec3::X, nan, Vec3::Z);
        assert!(p.is_finite(), "NaN ray origin produced {p:?}");
        let p = closest_on_line(Vec3::ZERO, Vec3::X, Vec3::Z * -5.0, nan);
        assert!(p.is_finite(), "NaN ray direction produced {p:?}");

        // `ray_plane` refuses rather than returning Some(NaN).
        assert!(ray_plane(Vec3::ZERO, Vec3::X, Vec3::ZERO, Vec3::Y).is_none());
        assert!(ray_plane(nan, Vec3::Z, Vec3::ZERO, Vec3::Y).is_none());
        assert!(ray_plane(Vec3::ZERO, nan, Vec3::ZERO, Vec3::Y).is_none());
        assert!(ray_plane(Vec3::ZERO, Vec3::Z, Vec3::ZERO, nan).is_none());
        // The healthy intersection is unchanged.
        let hit = ray_plane(Vec3::new(0.0, 5.0, 0.0), -Vec3::Y, Vec3::ZERO, Vec3::Y).unwrap();
        assert!((hit - Vec3::ZERO).length() < 1e-5, "{hit:?}");
    }

    #[test]
    fn screen_size_grows_with_distance() {
        let near = gizmo_world_size(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 5.0), 1.0);
        let far = gizmo_world_size(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 50.0), 1.0);
        assert!(far > near * 9.0 && far < near * 11.0);
    }

    #[test]
    fn translate_drag_constrains_to_axis() {
        // Origin at 0; grab down +X. Cursor ray moving diagonally should only
        // produce an X translation.
        let origin = Vec3::ZERO;
        // Ray from above looking down -Y, positioned over x=2.
        let drag = GizmoDrag::begin(
            GizmoMode::Translate,
            GizmoAxis::X,
            Quat::IDENTITY,
            origin,
            Vec3::new(2.0, 5.0, 0.0),
            Vec3::NEG_Y,
        );
        // Move the ray to x=5.
        let d = drag.update(Vec3::new(5.0, 5.0, 0.0), Vec3::NEG_Y, 0.0);
        match d {
            GizmoDelta::Translate(t) => {
                assert!((t.x - 3.0).abs() < 1e-3, "x delta {t:?}");
                assert!(t.y.abs() < 1e-6 && t.z.abs() < 1e-6, "off-axis {t:?}");
            }
            _ => panic!("expected translate"),
        }
    }

    #[test]
    fn translate_snaps_to_grid() {
        let origin = Vec3::ZERO;
        let drag = GizmoDrag::begin(
            GizmoMode::Translate,
            GizmoAxis::X,
            Quat::IDENTITY,
            origin,
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::NEG_Y,
        );
        let d = drag.update(Vec3::new(2.3, 5.0, 0.0), Vec3::NEG_Y, 1.0);
        match d {
            GizmoDelta::Translate(t) => assert!((t.x - 2.0).abs() < 1e-6, "snap {t:?}"),
            _ => panic!(),
        }
    }

    /// Extract the X component of a translate delta (test helper).
    fn tx(d: GizmoDelta) -> f64 {
        match d {
            GizmoDelta::Translate(t) => t.x,
            _ => panic!("expected translate"),
        }
    }

    #[test]
    fn translate_snap_is_cumulative_not_per_frame() {
        // The host no longer re-anchors the drag each frame (M2): update() is
        // called repeatedly on the SAME drag and measures from the original grab,
        // so snapping quantizes the CUMULATIVE displacement. A slow sub-snap drag
        // that per-frame rounding would freeze at 0 forever instead crosses the
        // snap boundary and jumps exactly one step.
        let origin = Vec3::ZERO;
        let drag = GizmoDrag::begin(
            GizmoMode::Translate,
            GizmoAxis::X,
            Quat::IDENTITY,
            origin,
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::NEG_Y,
        );
        let snap = 1.0;
        // Cumulative 0.3 → below the half-step boundary → still 0.
        assert_eq!(
            tx(drag.update(Vec3::new(0.3, 5.0, 0.0), Vec3::NEG_Y, snap)),
            0.0
        );
        // Cumulative 0.4 (another sub-snap increment) → still 0.
        assert_eq!(
            tx(drag.update(Vec3::new(0.4, 5.0, 0.0), Vec3::NEG_Y, snap)),
            0.0
        );
        // Cumulative 0.6 → crosses 0.5 → snaps to exactly one step.
        assert_eq!(
            tx(drag.update(Vec3::new(0.6, 5.0, 0.0), Vec3::NEG_Y, snap)),
            1.0
        );
    }

    #[test]
    fn translate_snapped_total_is_multiple_of_step() {
        let origin = Vec3::ZERO;
        let drag = GizmoDrag::begin(
            GizmoMode::Translate,
            GizmoAxis::X,
            Quat::IDENTITY,
            origin,
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::NEG_Y,
        );
        let snap = 0.25;
        for &x in &[0.1_f32, 0.4, 0.7, 1.15, 2.02, 3.9] {
            let t = tx(drag.update(Vec3::new(x, 5.0, 0.0), Vec3::NEG_Y, snap));
            let steps = t / snap as f64;
            assert!(
                (steps - steps.round()).abs() < 1e-4,
                "x={x} → {t} is not a multiple of {snap}"
            );
        }
    }

    #[test]
    fn rotate_snapped_total_is_multiple_of_step() {
        // Cumulative rotate about Y; every snapped angle is a multiple of 15°.
        let origin = Vec3::ZERO;
        let step = 15f32.to_radians();
        let drag = GizmoDrag::begin(
            GizmoMode::Rotate,
            GizmoAxis::Y,
            Quat::IDENTITY,
            origin,
            Vec3::new(1.0, 5.0, 0.0),
            Vec3::NEG_Y,
        );
        for &(x, z) in &[(0.9_f32, -0.2_f32), (0.5, -0.5), (0.0, -1.0), (-0.7, -0.7)] {
            match drag.update(Vec3::new(x, 5.0, z), Vec3::NEG_Y, step) {
                GizmoDelta::Rotate { radians, .. } => {
                    let k = radians / step;
                    assert!(
                        (k - k.round()).abs() < 1e-3,
                        "angle {radians} is not a multiple of 15°"
                    );
                }
                _ => panic!("expected rotate"),
            }
        }
    }

    #[test]
    fn scale_snapped_total_is_multiple_of_step() {
        // Cumulative scale on X; every snapped ratio is a multiple of 0.1.
        let origin = Vec3::ZERO;
        let step = 0.1;
        let drag = GizmoDrag::begin(
            GizmoMode::Scale,
            GizmoAxis::X,
            Quat::IDENTITY,
            origin,
            Vec3::new(2.0, 5.0, 0.0),
            Vec3::NEG_Y,
        );
        for &x in &[2.1_f32, 2.5, 3.0, 4.3, 1.2] {
            match drag.update(Vec3::new(x, 5.0, 0.0), Vec3::NEG_Y, step) {
                GizmoDelta::Scale(s) => {
                    let k = s.x / step;
                    assert!(
                        (k - k.round()).abs() < 1e-3,
                        "scale {} is not a multiple of 0.1",
                        s.x
                    );
                }
                _ => panic!("expected scale"),
            }
        }
    }

    #[test]
    fn rotate_drag_measures_signed_angle() {
        // Rotate about Y. Grab at +X, move to +Z → -90° about Y (right-handed:
        // +X rotates toward -Z for +Y, so X→Z is negative).
        let origin = Vec3::ZERO;
        let drag = GizmoDrag::begin(
            GizmoMode::Rotate,
            GizmoAxis::Y,
            Quat::IDENTITY,
            origin,
            Vec3::new(1.0, 5.0, 0.0),
            Vec3::NEG_Y,
        );
        let d = drag.update(Vec3::new(0.0, 5.0, 1.0), Vec3::NEG_Y, 0.0);
        match d {
            GizmoDelta::Rotate { axis, radians } => {
                assert_eq!(axis, Vec3::Y);
                assert!((radians.abs() - std::f32::consts::FRAC_PI_2).abs() < 1e-3);
            }
            _ => panic!("expected rotate"),
        }
    }

    #[test]
    fn scale_drag_ratio_on_axis() {
        let origin = Vec3::ZERO;
        // Grab at x=2, move to x=4 → 2× on X only.
        let drag = GizmoDrag::begin(
            GizmoMode::Scale,
            GizmoAxis::X,
            Quat::IDENTITY,
            origin,
            Vec3::new(2.0, 5.0, 0.0),
            Vec3::NEG_Y,
        );
        let d = drag.update(Vec3::new(4.0, 5.0, 0.0), Vec3::NEG_Y, 0.0);
        match d {
            GizmoDelta::Scale(s) => {
                assert!((s.x - 2.0).abs() < 1e-3, "scale {s:?}");
                assert!((s.y - 1.0).abs() < 1e-6 && (s.z - 1.0).abs() < 1e-6);
            }
            _ => panic!("expected scale"),
        }
    }

    #[test]
    fn pick_axis_selects_nearest_handle() {
        // Look down -Z at the origin; project the X axis tip and click near it.
        let eye = Vec3::new(0.0, 0.0, 10.0);
        let view = glam::camera::rh::view::look_at_mat4(eye, Vec3::ZERO, Vec3::Y);
        let proj = glam::camera::rh::proj::directx::perspective(1.0, 16.0 / 9.0, 0.1, 100.0);
        let vp = proj * view;
        let (w, h) = (1600.0, 900.0);
        let size = 2.0;
        // Screen position of the +X tip.
        let tip = project(Vec3::X * size, vp, w, h).unwrap();
        let hit = pick_axis(
            GizmoMode::Translate,
            Vec3::ZERO,
            Quat::IDENTITY,
            size,
            vp,
            tip,
            w,
            h,
            false,
        );
        assert_eq!(hit, Some(GizmoAxis::X));
        // A click far away hits nothing.
        assert_eq!(
            pick_axis(
                GizmoMode::Translate,
                Vec3::ZERO,
                Quat::IDENTITY,
                size,
                vp,
                Vec2::new(5.0, 5.0),
                w,
                h,
                false,
            ),
            None
        );
    }

    #[test]
    fn build_geometry_emits_lines() {
        let mut d = DebugDraw::default();
        build_geometry(
            &mut d,
            GizmoMode::Translate,
            Vec3::ZERO,
            Quat::IDENTITY,
            1.0,
            Some(GizmoAxis::X),
            false,
        );
        assert!(!d.verts.is_empty());
        let mut r = DebugDraw::default();
        build_geometry(
            &mut r,
            GizmoMode::Rotate,
            Vec3::ZERO,
            Quat::IDENTITY,
            1.0,
            None,
            false,
        );
        assert!(r.verts.len() > 100); // three circles worth of segments
    }

    #[test]
    fn two_d_pick_excludes_z_axis() {
        // Look straight down -Z at the XY plane (the 2D editor view).
        let eye = Vec3::new(0.0, 0.0, 10.0);
        let view = glam::camera::rh::view::look_at_mat4(eye, Vec3::ZERO, Vec3::Y);
        let proj = glam::camera::rh::proj::directx::perspective(1.0, 16.0 / 9.0, 0.1, 100.0);
        let vp = proj * view;
        let (w, h) = (1600.0, 900.0);
        let size = 2.0;
        // The Z tip projects onto the origin (dead-on view); in 2D mode it must
        // never be picked — the nearby handle is the XY plane, not Z.
        let z_tip = project(Vec3::Z * size, vp, w, h).unwrap();
        let hit = pick_axis(
            GizmoMode::Translate,
            Vec3::ZERO,
            Quat::IDENTITY,
            size,
            vp,
            z_tip,
            w,
            h,
            true,
        );
        assert_ne!(hit, Some(GizmoAxis::Z), "Z axis must be inert in 2D");
        // The X tip still picks X.
        let x_tip = project(Vec3::X * size, vp, w, h).unwrap();
        assert_eq!(
            pick_axis(
                GizmoMode::Translate,
                Vec3::ZERO,
                Quat::IDENTITY,
                size,
                vp,
                x_tip,
                w,
                h,
                true
            ),
            Some(GizmoAxis::X)
        );
    }

    #[test]
    fn two_d_rotate_geometry_is_one_ring() {
        // 2D rotate is a single Z ring; 3D rotate is three rings — so the 2D
        // vertex count is about a third.
        let mut two = DebugDraw::default();
        build_geometry(
            &mut two,
            GizmoMode::Rotate,
            Vec3::ZERO,
            Quat::IDENTITY,
            1.0,
            None,
            true,
        );
        let mut three = DebugDraw::default();
        build_geometry(
            &mut three,
            GizmoMode::Rotate,
            Vec3::ZERO,
            Quat::IDENTITY,
            1.0,
            None,
            false,
        );
        assert!(!two.verts.is_empty());
        assert!(two.verts.len() < three.verts.len() / 2);
    }

    #[test]
    fn local_basis_translate_follows_rotated_axis() {
        // A 90° yaw about +Y maps the local +X handle onto world -Z. Dragging
        // that handle must translate along world -Z (its length), with no world-X
        // component — proving the basis rotates the constraint axis.
        let origin = Vec3::ZERO;
        let basis = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2); // +X → -Z
                                                                        // Ray from above (looking down -Y) grabbing the rotated X handle at z=-2.
        let drag = GizmoDrag::begin(
            GizmoMode::Translate,
            GizmoAxis::X,
            basis,
            origin,
            Vec3::new(0.0, 5.0, -2.0),
            Vec3::NEG_Y,
        );
        let d = drag.update(Vec3::new(0.0, 5.0, -5.0), Vec3::NEG_Y, 0.0);
        match d {
            GizmoDelta::Translate(t) => {
                assert!(t.x.abs() < 1e-3, "should not move in world X: {t:?}");
                assert!((t.z + 3.0).abs() < 1e-3, "should move -3 in world Z: {t:?}");
                assert!(t.y.abs() < 1e-6, "off-axis {t:?}");
            }
            _ => panic!("expected translate"),
        }
    }

    #[test]
    fn local_basis_rotate_is_about_rotated_axis() {
        // With a basis that maps local +X onto world +Y, a rotate drag on the X
        // ring must report its rotation axis as world +Y (the basis-rotated axis).
        let origin = Vec3::ZERO;
        let basis = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2); // +X → +Y
                                                                        // The X ring's world normal is +Y, so it lies in the XZ plane; grab/move
                                                                        // it with rays coming straight down -Y (which cross that plane).
        let drag = GizmoDrag::begin(
            GizmoMode::Rotate,
            GizmoAxis::X,
            basis,
            origin,
            Vec3::new(1.0, 5.0, 0.0),
            Vec3::NEG_Y,
        );
        match drag.update(Vec3::new(0.0, 5.0, 1.0), Vec3::NEG_Y, 0.0) {
            GizmoDelta::Rotate { axis, radians } => {
                assert!(
                    (axis - Vec3::Y).length() < 1e-5,
                    "rotate axis should be world +Y, got {axis:?}"
                );
                assert!(radians.abs() > 1e-3, "a real rotation should be measured");
            }
            _ => panic!("expected rotate"),
        }
    }

    #[test]
    fn world_basis_matches_pre_basis_behaviour() {
        // Regression: an identity basis must reproduce the exact world-space
        // delta the gizmo produced before basis threading landed.
        let origin = Vec3::ZERO;
        let drag = GizmoDrag::begin(
            GizmoMode::Translate,
            GizmoAxis::X,
            Quat::IDENTITY,
            origin,
            Vec3::new(2.0, 5.0, 0.0),
            Vec3::NEG_Y,
        );
        let d = drag.update(Vec3::new(5.0, 5.0, 0.0), Vec3::NEG_Y, 0.0);
        match d {
            GizmoDelta::Translate(t) => {
                assert!((t.x - 3.0).abs() < 1e-3, "x delta {t:?}");
                assert!(t.y.abs() < 1e-6 && t.z.abs() < 1e-6, "off-axis {t:?}");
            }
            _ => panic!("expected translate"),
        }
    }
}
