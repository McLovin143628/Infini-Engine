//! View math: f64 world eye → f32 render-local matrices (reverse-infinite Z).
//!
//! The renderer is camera-*policy*-free: editor camera behavior (flycam,
//! orbit, focus) lives in the host. A [`RenderView`] is the resolved per-frame
//! snapshot: floating origin, world-space eye, orientation, projection.

use glam::{DVec3, Mat4, Vec3, Vec4};
use inf_math::FloatingOrigin;

/// Reverse-Z everywhere: clear depth to 0.0, depth compare = Greater.
/// Reversed infinite projection gives near-constant depth precision across
/// the whole range — the production choice for large worlds.
pub const DEPTH_CLEAR: f32 = 0.0;
pub const DEPTH_COMPARE: wgpu::CompareFunction = wgpu::CompareFunction::Greater;
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Orthographic projection parameters (2D editor mode). Reverse-Z over a
/// *finite* `[near, far]` range (see [`ortho_reverse_z`]); when present on a
/// [`RenderView`] it replaces the perspective projection while every other pass
/// (grid, sky, meshes, sprites, debug, picking, gizmos) keeps working — depth
/// still clears to 0 and compares `Greater`.
#[derive(Debug, Clone, Copy)]
pub struct OrthoParams {
    /// Half the visible world-space height; the width follows from the aspect
    /// ratio. Smaller = more zoomed in.
    pub half_height: f32,
    /// View-space near/far distances ahead of the eye (both positive,
    /// `near < far`). Near maps to depth 1.0, far to depth 0.0 (reverse-Z).
    pub near: f32,
    pub far: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct RenderView {
    pub origin: FloatingOrigin,
    pub eye_world: DVec3,
    /// Unit view direction (render-local == world axes; the origin only
    /// translates).
    pub forward: Vec3,
    pub up: Vec3,
    /// Vertical field of view, radians. Ignored when [`Self::ortho`] is set.
    pub fov_y: f32,
    pub near: f32,
    /// Scene render size in physical pixels.
    pub width: u32,
    pub height: u32,
    /// `Some` ⇒ orthographic (2D editor mode); `None` ⇒ perspective
    /// reverse-infinite-Z (the 3D default). Kept `Option` so every existing
    /// perspective call-site and golden stays byte-identical.
    pub ortho: Option<OrthoParams>,
}

/// Reverse-Z orthographic projection for a right-handed view space (looking
/// down its own -Z), producing wgpu clip space (NDC z ∈ [0, 1], y up). The
/// near plane maps to depth **1.0** and the far plane to **0.0** — the same
/// convention as [`perspective_infinite_reverse`], so `DEPTH_CLEAR` (0.0) and
/// `DEPTH_COMPARE` (`Greater`) drive every depth-testing pass unchanged in 2D.
///
/// [`perspective_infinite_reverse`]: glam::camera::rh::proj::directx::perspective_infinite_reverse
pub fn ortho_reverse_z(half_height: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let half_w = (half_height * aspect).max(1e-6);
    let hh = half_height.max(1e-6);
    let inv_range = 1.0 / (far - near).max(1e-6);
    // Column-major (glam): clip.z = ze·inv_range + far·inv_range, clip.w = 1.
    Mat4::from_cols(
        Vec4::new(1.0 / half_w, 0.0, 0.0, 0.0),
        Vec4::new(0.0, 1.0 / hh, 0.0, 0.0),
        Vec4::new(0.0, 0.0, inv_range, 0.0),
        Vec4::new(0.0, 0.0, far * inv_range, 1.0),
    )
}

impl RenderView {
    pub fn eye_local(&self) -> Vec3 {
        self.origin.to_render(self.eye_world)
    }

    pub fn aspect(&self) -> f32 {
        self.width.max(1) as f32 / self.height.max(1) as f32
    }

    pub fn view_proj(&self) -> Mat4 {
        let view = glam::camera::rh::view::look_to_mat4(self.eye_local(), self.forward, self.up);
        let proj = match self.ortho {
            None => glam::camera::rh::proj::directx::perspective_infinite_reverse(
                self.fov_y,
                self.aspect(),
                self.near,
            ),
            Some(o) => ortho_reverse_z(o.half_height, self.aspect(), o.near, o.far),
        };
        proj * view
    }

    /// Ray through a pixel (physical px, origin top-left), in render-local
    /// space. Returns (ray origin, unit direction).
    pub fn pixel_ray(&self, px: f32, py: f32) -> (Vec3, Vec3) {
        let ndc_x = px / self.width.max(1) as f32 * 2.0 - 1.0;
        let ndc_y = 1.0 - py / self.height.max(1) as f32 * 2.0;
        let inv = self.view_proj().inverse();
        // With an infinite-reverse projection the far plane (depth 0) is at
        // infinity; unproject two finite depths instead and extend.
        let a = inv.project_point3(Vec3::new(ndc_x, ndc_y, 0.9));
        let b = inv.project_point3(Vec3::new(ndc_x, ndc_y, 0.4));
        let dir = (b - a).normalize_or_zero();
        let origin = if self.ortho.is_some() {
            // Orthographic: rays are parallel (all share `dir`); the origin is
            // the point on the near plane under the pixel (depth 1.0 = near in
            // reverse-Z), not the eye.
            inv.project_point3(Vec3::new(ndc_x, ndc_y, 1.0))
        } else {
            self.eye_local()
        };
        (origin, dir)
    }
}

/// GPU-side view uniforms, shared by every pass (bind group 0). Layout must
/// match `struct View` in the WGSL shaders.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ViewUniforms {
    pub view_proj: [f32; 16],
    pub inv_view_proj: [f32; 16],
    /// xyz = eye position (render-local), w unused.
    pub eye: [f32; 4],
    /// xyz = unit direction TOWARD the sun, w unused.
    pub sun_dir: [f32; 4],
    /// x = -origin.x, y = -origin.z as f32 (render-local position of the
    /// world X/Z axes for the grid shader), zw = viewport size in px.
    pub grid_axis_viewport: [f32; 4],
    /// x = 1.0 in orthographic (2D) mode else 0.0; y = -origin.y as f32
    /// (render-local position of the world Y axis, for the 2D XY grid);
    /// zw reserved. Appended so pre-2D passes/goldens stay byte-identical.
    pub mode_axis: [f32; 4],
    /// Camera **right** basis vector (render-local == world axes; the origin only
    /// translates), xyz; w unused. Read only by the sprite pass to orient
    /// billboards (P8.4a). Appended after `mode_axis` so every other pass — which
    /// declares the shorter `View` struct — is unaffected and its goldens stay
    /// byte-identical (a uniform buffer may be larger than a shader's struct).
    pub cam_right: [f32; 4],
    /// Camera **up** basis vector (render-local), xyz; w unused. See `cam_right`.
    pub cam_up: [f32; 4],
    /// View-mode flags (R-P2): `x` = unlit (1.0 ⇒ the lit scene passes return
    /// albedo+emissive directly, skipping the light loop — drives both the Unlit
    /// and Wireframe view modes); `yzw` reserved. Appended last so every pass that
    /// declares the shorter `View` struct is unaffected and every pre-R-P2 golden
    /// stays byte-identical (the renderer writes 0 here for the default Lit mode).
    pub flags: [f32; 4],
}

/// Fixed editor sun for Phase 2 (light theory arrives with materials, P7).
pub const SUN_DIR: Vec3 = Vec3::new(0.45, 0.75, 0.3);

impl ViewUniforms {
    pub fn from_view(view: &RenderView) -> Self {
        let vp = view.view_proj();
        let origin = view.origin.origin();
        // Camera basis (render-local == world directions): right = fwd × up,
        // true up = right × fwd. Used to orient billboard sprites.
        let fwd = view.forward.normalize_or_zero();
        let cam_right = fwd.cross(view.up).normalize_or_zero();
        let cam_up = cam_right.cross(fwd).normalize_or_zero();
        Self {
            view_proj: vp.to_cols_array(),
            inv_view_proj: vp.inverse().to_cols_array(),
            eye: view.eye_local().extend(0.0).to_array(),
            sun_dir: SUN_DIR.normalize().extend(0.0).to_array(),
            grid_axis_viewport: Vec4::new(
                -origin.x as f32,
                -origin.z as f32,
                view.width as f32,
                view.height as f32,
            )
            .to_array(),
            mode_axis: Vec4::new(
                if view.ortho.is_some() { 1.0 } else { 0.0 },
                -origin.y as f32,
                0.0,
                0.0,
            )
            .to_array(),
            cam_right: cam_right.extend(0.0).to_array(),
            cam_up: cam_up.extend(0.0).to_array(),
            // Default Lit: no unlit flag. The renderer overwrites `flags[0]` per
            // frame from its active view mode (Unlit/Wireframe set it to 1.0).
            flags: [0.0; 4],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_view() -> RenderView {
        RenderView {
            origin: FloatingOrigin::new(DVec3::ZERO),
            eye_world: DVec3::new(0.0, 2.0, 5.0),
            forward: Vec3::NEG_Z,
            up: Vec3::Y,
            fov_y: 60f32.to_radians(),
            near: 0.05,
            width: 1600,
            height: 900,
            ortho: None,
        }
    }

    /// A top-down orthographic view over the XY plane (2D editor mode): eye at
    /// +Z looking down -Z, up = +Y, half-height 5 world units.
    fn ortho_view() -> RenderView {
        RenderView {
            origin: FloatingOrigin::new(DVec3::ZERO),
            eye_world: DVec3::new(0.0, 0.0, 100.0),
            forward: Vec3::NEG_Z,
            up: Vec3::Y,
            fov_y: 60f32.to_radians(),
            near: 1.0,
            width: 1600,
            height: 900,
            ortho: Some(OrthoParams {
                half_height: 5.0,
                near: 1.0,
                far: 200.0,
            }),
        }
    }

    #[test]
    fn center_pixel_ray_matches_forward() {
        let v = test_view();
        let (o, d) = v.pixel_ray(800.0, 450.0);
        assert!((o - Vec3::new(0.0, 2.0, 5.0)).length() < 1e-4);
        assert!(d.dot(v.forward) > 0.9999, "ray {d:?} not along forward");
    }

    #[test]
    fn reverse_z_orders_depth() {
        let v = test_view();
        let vp = v.view_proj();
        let near_p = vp.project_point3(Vec3::new(0.0, 2.0, 4.0)); // 1 m ahead
        let far_p = vp.project_point3(Vec3::new(0.0, 2.0, -95.0)); // 100 m ahead
                                                                   // Reverse-Z: nearer surfaces have LARGER depth values.
        assert!(near_p.z > far_p.z);
        assert!(far_p.z > 0.0 && near_p.z <= 1.0);
    }

    #[test]
    fn pixel_ray_hits_known_ground_point() {
        // Eye 2 m up looking straight down -Z: the ray through the pixel of
        // ground point (0, 0, -2) must hit y=0 at z=-2.
        let v = test_view();
        let vp = v.view_proj();
        let clip =
            vp.project_point3(Vec3::new(0.0, 0.0, -2.0) - Vec3::new(0.0, 2.0, 5.0) + v.eye_local());
        let px = (clip.x * 0.5 + 0.5) * v.width as f32;
        let py = (0.5 - clip.y * 0.5) * v.height as f32;
        let (o, d) = v.pixel_ray(px, py);
        let t = -o.y / d.y;
        let hit = o + d * t;
        assert!(
            (hit - Vec3::new(0.0, 0.0, -2.0)).length() < 1e-2,
            "hit {hit:?}"
        );
    }

    #[test]
    fn uniforms_pack_grid_axis_from_origin() {
        let mut v = test_view();
        v.origin = FloatingOrigin::new(DVec3::new(500.0, 0.0, -300.0));
        let u = ViewUniforms::from_view(&v);
        assert_eq!(u.grid_axis_viewport[0], -500.0);
        assert_eq!(u.grid_axis_viewport[1], 300.0);
        assert_eq!(
            std::mem::size_of::<ViewUniforms>(),
            (16 + 16 + 4 + 4 + 4 + 4 + 4 + 4 + 4) * 4
        );
        // The view-mode flags default to Lit (all zero) — this is what keeps every
        // pre-R-P2 golden byte-identical (the shaders branch only on `flags.x`).
        assert_eq!(u.flags, [0.0; 4]);
        // Perspective packs mode flag 0; the origin's Y still populates the 2D
        // axis slot (only read by the grid shader in ortho).
        assert_eq!(u.mode_axis[0], 0.0);
    }

    #[test]
    fn uniforms_pack_camera_basis_for_billboards() {
        // A camera looking down -Z with up +Y: right = +X, up = +Y.
        let v = test_view();
        let u = ViewUniforms::from_view(&v);
        assert!((Vec3::from_slice(&u.cam_right[..3]) - Vec3::X).length() < 1e-6);
        assert!((Vec3::from_slice(&u.cam_up[..3]) - Vec3::Y).length() < 1e-6);
    }

    #[test]
    fn ortho_projects_ndc_corners() {
        let v = ortho_view();
        let vp = v.view_proj();
        let hh = 5.0_f32;
        let hw = hh * v.aspect();
        // A world point one half-extent right and up of the eye's XY lands at
        // the +X/+Y NDC corner; the eye's XY lands at NDC center.
        let center = vp.project_point3(Vec3::new(0.0, 0.0, 0.0));
        assert!(center.x.abs() < 1e-4 && center.y.abs() < 1e-4, "{center:?}");
        let corner = vp.project_point3(Vec3::new(hw, hh, 0.0));
        assert!((corner.x - 1.0).abs() < 1e-4, "ndc x {corner:?}");
        assert!((corner.y - 1.0).abs() < 1e-4, "ndc y {corner:?}");
    }

    #[test]
    fn ortho_reverse_z_orders_depth() {
        let v = ortho_view();
        let vp = v.view_proj();
        // Nearer to the eye (larger Z, since eye is at +Z looking -Z) ⇒ LARGER
        // depth under reverse-Z, and depths stay within [0, 1].
        let near_p = vp.project_point3(Vec3::new(0.0, 0.0, 10.0)); // dist 90
        let far_p = vp.project_point3(Vec3::new(0.0, 0.0, -10.0)); // dist 110
        assert!(near_p.z > far_p.z, "near {near_p:?} far {far_p:?}");
        assert!(far_p.z >= 0.0 && near_p.z <= 1.0);
    }

    #[test]
    fn ortho_pixel_ray_is_parallel_and_hits_plane() {
        let v = ortho_view();
        // Two different pixels give parallel rays (both point along -Z).
        let (o0, d0) = v.pixel_ray(400.0, 300.0);
        let (o1, d1) = v.pixel_ray(1200.0, 700.0);
        assert!(d0.dot(Vec3::NEG_Z) > 0.9999, "ray0 {d0:?}");
        assert!((d0 - d1).length() < 1e-4, "rays not parallel");
        // The two ray origins differ (parallel projection: origin varies with
        // the pixel, unlike perspective where all rays share the eye).
        assert!((o0 - o1).length() > 1.0);
        // Each ray hits the XY plane (z = 0) in front of its near-plane origin.
        let t = -o0.z / d0.z;
        assert!(t > 0.0, "plane behind origin");
    }
}
