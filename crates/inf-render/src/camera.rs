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

#[derive(Debug, Clone, Copy)]
pub struct RenderView {
    pub origin: FloatingOrigin,
    pub eye_world: DVec3,
    /// Unit view direction (render-local == world axes; the origin only
    /// translates).
    pub forward: Vec3,
    pub up: Vec3,
    /// Vertical field of view, radians.
    pub fov_y: f32,
    pub near: f32,
    /// Scene render size in physical pixels.
    pub width: u32,
    pub height: u32,
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
        let proj = glam::camera::rh::proj::directx::perspective_infinite_reverse(
            self.fov_y,
            self.aspect(),
            self.near,
        );
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
        (self.eye_local(), dir)
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
}

/// Fixed editor sun for Phase 2 (light theory arrives with materials, P7).
pub const SUN_DIR: Vec3 = Vec3::new(0.45, 0.75, 0.3);

impl ViewUniforms {
    pub fn from_view(view: &RenderView) -> Self {
        let vp = view.view_proj();
        let origin = view.origin.origin();
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
            (16 + 16 + 4 + 4 + 4) * 4
        );
    }
}
