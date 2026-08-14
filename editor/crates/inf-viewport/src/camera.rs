//! Editor camera controller: pure input→state math, platform-neutral and
//! unit-tested headless (the per-OS hosts only feed it input deltas).
//!
//! UE-parity navigation:
//! - RMB captured: mouse look + WASD/QE fly, wheel scales speed, Shift boosts.
//! - Alt+LMB orbit / Alt+MMB (or plain MMB) pan / Alt+RMB dolly around a pivot.
//! - Wheel dollies toward the look point when not flying.
//! - F focuses the selection (smooth interpolation); Ctrl+1..9 store camera
//!   bookmarks, 1..9 recall them.
//!
//! The camera is a yaw/pitch/eye triple; orbit/focus derive a pivot on demand
//! rather than storing one, so switching between fly and orbit is seamless.

use glam::{DVec2, DVec3, Vec3};

/// Radians per raw mouse count.
const LOOK_SENSITIVITY: f32 = 0.0032;
/// Orbit uses the same angular sensitivity as free-look.
const ORBIT_SENSITIVITY: f32 = 0.0060;
/// Pitch stays just short of the poles to keep `forward` well-defined.
const PITCH_LIMIT: f32 = 1.55;
pub const FLY_SPEED_MIN: f32 = 0.2;
pub const FLY_SPEED_MAX: f32 = 250.0;
/// Fraction of the remaining focus distance covered per second (exponential
/// ease-out). ~0.001 of the gap remains after the time-constant.
const FOCUS_RATE: f32 = 12.0;
/// Focus interpolation snaps to done inside this world-space distance.
const FOCUS_EPSILON: f64 = 0.01;
/// Camera bookmark slots (recalled by digit keys 1..=9).
pub const BOOKMARK_SLOTS: usize = 9;
/// Default pivot distance ahead of the eye when nothing is selected.
const DEFAULT_PIVOT_DISTANCE: f64 = 10.0;

/// Accumulated flycam input for one frame (already coalesced by the host).
#[derive(Debug, Clone, Copy, Default)]
pub struct FlyInput {
    /// Raw mouse deltas while captured.
    pub mouse_dx: f32,
    pub mouse_dy: f32,
    /// Wheel detents (speed scaling while captured).
    pub wheel_steps: i32,
    pub forward: bool,
    pub back: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
    pub boost: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct EditorCamera {
    /// World-space eye (f64 — architecture rule 3).
    pub pos: DVec3,
    /// Radians around +Y; 0 looks down -Z.
    pub yaw: f32,
    /// Radians; positive looks up.
    pub pitch: f32,
    /// Metres per second while flying.
    pub fly_speed: f32,
}

impl Default for EditorCamera {
    fn default() -> Self {
        // Perched behind-right of the origin, overlooking the demo field.
        Self {
            pos: DVec3::new(14.0, 9.0, 20.0),
            yaw: -0.55,
            pitch: -0.35,
            fly_speed: 8.0,
        }
    }
}

impl EditorCamera {
    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(sy * cp, sp, -cy * cp)
    }

    pub fn right(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        Vec3::new(cy, 0.0, sy)
    }

    /// One frame of captured flycam movement.
    pub fn apply_fly(&mut self, input: &FlyInput, dt: f32) {
        if input.wheel_steps != 0 {
            self.fly_speed = (self.fly_speed * 1.2f32.powi(input.wheel_steps))
                .clamp(FLY_SPEED_MIN, FLY_SPEED_MAX);
        }

        self.yaw += input.mouse_dx * LOOK_SENSITIVITY;
        self.pitch =
            (self.pitch - input.mouse_dy * LOOK_SENSITIVITY).clamp(-PITCH_LIMIT, PITCH_LIMIT);

        let mut mv = Vec3::ZERO;
        if input.forward {
            mv += self.forward();
        }
        if input.back {
            mv -= self.forward();
        }
        if input.right {
            mv += self.right();
        }
        if input.left {
            mv -= self.right();
        }
        if input.up {
            mv += Vec3::Y;
        }
        if input.down {
            mv -= Vec3::Y;
        }
        if mv != Vec3::ZERO {
            let boost = if input.boost { 4.0 } else { 1.0 };
            let step = mv.normalize() * self.fly_speed * boost * dt;
            self.pos += step.as_dvec3();
        }
    }

    fn set_forward(&mut self, dir: Vec3) {
        let d = dir.normalize_or_zero();
        if d == Vec3::ZERO {
            return;
        }
        self.pitch = d.y.clamp(-1.0, 1.0).asin().clamp(-PITCH_LIMIT, PITCH_LIMIT);
        self.yaw = d.x.atan2(-d.z);
    }

    /// Point the camera at `target` from its current position.
    pub fn look_at(&mut self, target: DVec3) {
        self.set_forward((target - self.pos).as_vec3());
    }

    /// The point the camera orbits/dollies around: the caller's focus point
    /// (selection center) if any, else a fixed distance down the view ray.
    pub fn pivot(&self, focus: Option<DVec3>) -> DVec3 {
        focus.unwrap_or_else(|| self.pos + self.forward().as_dvec3() * DEFAULT_PIVOT_DISTANCE)
    }

    /// One frame of orbit/pan/dolly around `pivot` (Maya/UE Alt-navigation).
    pub fn apply_navigate(&mut self, input: &NavInput, pivot: DVec3, dt: f32) {
        let _ = dt; // navigation is velocity-free (direct mouse mapping)
        match input.mode {
            NavMode::Orbit => {
                let radius = (self.pos - pivot).length();
                self.yaw += input.mouse_dx * ORBIT_SENSITIVITY;
                self.pitch = (self.pitch + input.mouse_dy * ORBIT_SENSITIVITY)
                    .clamp(-PITCH_LIMIT, PITCH_LIMIT);
                // Re-place the eye on the sphere, still looking at the pivot.
                self.pos = pivot - self.forward().as_dvec3() * radius.max(0.05);
            }
            NavMode::Pan => {
                // Screen-space pan scaled by pivot distance so it tracks the
                // point under the cursor regardless of zoom.
                let dist = (self.pos - pivot).length() as f32;
                let scale = dist * 0.0016;
                let delta =
                    -self.right() * input.mouse_dx * scale + self.up() * input.mouse_dy * scale;
                self.pos += delta.as_dvec3();
            }
            NavMode::Dolly => {
                // Horizontal drag / wheel moves along the view ray toward the
                // pivot; never crosses it.
                self.dolly(
                    input.mouse_dx * 0.01 + input.wheel_steps as f32 * 0.12,
                    pivot,
                );
            }
            NavMode::None => {}
        }
    }

    fn up(&self) -> Vec3 {
        self.right().cross(self.forward()).normalize_or_zero()
    }

    /// Move along the view ray by `amount` (fraction of pivot distance);
    /// positive pulls toward the pivot. Clamped so the eye never passes it.
    pub fn dolly(&mut self, amount: f32, pivot: DVec3) {
        let to_pivot = pivot - self.pos;
        let dist = to_pivot.length();
        if dist < 1e-4 {
            return;
        }
        let step = (dist as f32 * amount).clamp(-1e6, dist as f32 - 0.05) as f64;
        self.pos += to_pivot / dist * step;
    }

    /// Begin a smooth focus so `target` fills a `radius`-metre sphere. Returns
    /// the goal pose; feed [`Self::advance_focus`] each frame until settled.
    pub fn focus_goal(&self, target: DVec3, radius: f64) -> CameraPose {
        // Back off far enough for the whole radius to fit a ~60° fov, keeping
        // the current view direction.
        let dir = self.forward().as_dvec3();
        let dist = (radius / (30f32.to_radians().tan() as f64)).max(radius * 1.5 + 0.5);
        CameraPose {
            pos: target - dir * dist,
            yaw: self.yaw,
            pitch: self.pitch,
        }
    }

    /// Exponentially ease pos/yaw/pitch toward `goal`. Returns true once the
    /// camera has effectively arrived (caller stops advancing).
    pub fn advance_focus(&mut self, goal: &CameraPose, dt: f32) -> bool {
        let t = 1.0 - (-FOCUS_RATE * dt).exp();
        self.pos = self.pos.lerp(goal.pos, t as f64);
        self.yaw = lerp_angle(self.yaw, goal.yaw, t);
        self.pitch += (goal.pitch - self.pitch) * t;
        if self.pos.distance(goal.pos) < FOCUS_EPSILON {
            self.pos = goal.pos;
            self.yaw = goal.yaw;
            self.pitch = goal.pitch;
            return true;
        }
        false
    }

    pub fn pose(&self) -> CameraPose {
        CameraPose {
            pos: self.pos,
            yaw: self.yaw,
            pitch: self.pitch,
        }
    }

    pub fn set_pose(&mut self, pose: CameraPose) {
        self.pos = pose.pos;
        self.yaw = pose.yaw;
        self.pitch = pose.pitch;
    }
}

/// Which Alt-navigation gesture is active this frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NavMode {
    #[default]
    None,
    Orbit,
    Pan,
    Dolly,
}

/// Accumulated orbit/pan/dolly input for one frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct NavInput {
    pub mode: NavMode,
    pub mouse_dx: f32,
    pub mouse_dy: f32,
    pub wheel_steps: i32,
}

/// A restorable camera pose (bookmarks, focus goals).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraPose {
    pub pos: DVec3,
    pub yaw: f32,
    pub pitch: f32,
}

/// Nine camera bookmark slots recalled by the digit keys.
#[derive(Debug, Clone, Copy, Default)]
pub struct Bookmarks {
    slots: [Option<CameraPose>; BOOKMARK_SLOTS],
}

impl Bookmarks {
    /// Store `pose` in slot `n` (1-based digit key). Out-of-range is a no-op.
    pub fn store(&mut self, n: usize, pose: CameraPose) {
        if (1..=BOOKMARK_SLOTS).contains(&n) {
            self.slots[n - 1] = Some(pose);
        }
    }

    /// Recall slot `n` (1-based), or `None` if empty/out-of-range.
    pub fn recall(&self, n: usize) -> Option<CameraPose> {
        if (1..=BOOKMARK_SLOTS).contains(&n) {
            self.slots[n - 1]
        } else {
            None
        }
    }
}

/// Which projection the viewport is driving (P8.2c). Each mode keeps its own
/// camera state so switching back and forth restores the exact pose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ViewportMode {
    #[default]
    Perspective,
    /// Orthographic top-down 2D editing over the XY plane.
    TwoD,
}

/// Eye height above the XY plane in 2D mode (world units). Large enough that the
/// ortho near/far bracket typical sprite Z, small enough to keep f32 precision.
pub const TWO_D_EYE_HEIGHT: f64 = 100.0;
/// Ortho view-space near/far (see `inf_render::OrthoParams`): the world XY plane
/// sits at distance `TWO_D_EYE_HEIGHT`, comfortably inside `[near, far]`.
pub const TWO_D_NEAR: f32 = 1.0;
pub const TWO_D_FAR: f32 = 200.0;
/// Zoom (half-height) clamps in world units.
pub const TWO_D_HALF_HEIGHT_MIN: f64 = 0.05;
pub const TWO_D_HALF_HEIGHT_MAX: f64 = 5000.0;
/// Exponential zoom factor per wheel detent.
const TWO_D_ZOOM_STEP: f64 = 1.2;

/// Orthographic 2D editor camera: looks straight down -Z onto the world XY
/// plane, up = +Y, roll locked. Pan moves `center`; the wheel scales
/// `half_height` about the cursor. World coordinates are f64 (architecture
/// rule 3); the eye still rebases through the floating origin in the host.
#[derive(Debug, Clone, Copy)]
pub struct Camera2D {
    /// World-space XY point the viewport is centered on.
    pub center: DVec2,
    /// Half the visible world-space height (zoom); smaller = more zoomed in.
    pub half_height: f64,
}

impl Default for Camera2D {
    fn default() -> Self {
        Self {
            center: DVec2::ZERO,
            half_height: 8.0,
        }
    }
}

impl Camera2D {
    /// World-space eye position: fixed height above the plane over `center`.
    pub fn eye(&self) -> DVec3 {
        DVec3::new(self.center.x, self.center.y, TWO_D_EYE_HEIGHT)
    }

    fn half_width(&self, aspect: f64) -> f64 {
        self.half_height * aspect
    }

    /// World XY point under a viewport pixel (origin top-left).
    pub fn world_at_pixel(&self, px: f64, py: f64, width: f64, height: f64) -> DVec2 {
        let (w, h) = (width.max(1.0), height.max(1.0));
        let aspect = w / h;
        let nx = px / w * 2.0 - 1.0;
        let ny = 1.0 - py / h * 2.0;
        DVec2::new(
            self.center.x + nx * self.half_width(aspect),
            self.center.y + ny * self.half_height,
        )
    }

    /// Zoom about the cursor: scale `half_height` exponentially by the wheel
    /// steps (scroll up = zoom in) and shift `center` so the world point under
    /// the cursor stays fixed — the zoom-to-cursor invariant.
    pub fn zoom_at(&mut self, wheel_steps: i32, px: f64, py: f64, width: f64, height: f64) {
        if wheel_steps == 0 {
            return;
        }
        let (w, h) = (width.max(1.0), height.max(1.0));
        let aspect = w / h;
        let nx = px / w * 2.0 - 1.0;
        let ny = 1.0 - py / h * 2.0;
        let hh_old = self.half_height;
        let hh_new = (hh_old * TWO_D_ZOOM_STEP.powi(-wheel_steps))
            .clamp(TWO_D_HALF_HEIGHT_MIN, TWO_D_HALF_HEIGHT_MAX);
        let delta = hh_old - hh_new;
        self.center.x += nx * aspect * delta;
        self.center.y += ny * delta;
        self.half_height = hh_new;
    }

    /// Pan by a pixel drag (MMB/RMB): move `center` opposite the drag so the
    /// grabbed point tracks the cursor.
    pub fn pan(&mut self, dx: f64, dy: f64, width: f64, height: f64) {
        let (w, h) = (width.max(1.0), height.max(1.0));
        let aspect = w / h;
        let wx_per_px = 2.0 * self.half_width(aspect) / w;
        let wy_per_px = 2.0 * self.half_height / h;
        self.center.x -= dx * wx_per_px;
        self.center.y += dy * wy_per_px;
    }

    /// Frame an XY region (F): center on `center` and zoom so a
    /// `half_extent` (world half-size in XY) box fits with padding. `aspect` =
    /// width / height.
    pub fn frame(&mut self, center: DVec2, half_extent: DVec2, aspect: f64) {
        self.center = center;
        let pad = 1.3;
        let need_h = half_extent.y.max(half_extent.x / aspect.max(1e-3)).max(0.5) * pad;
        self.half_height = need_h.clamp(TWO_D_HALF_HEIGHT_MIN, TWO_D_HALF_HEIGHT_MAX);
    }
}

/// 2D-mode snapping configuration, pushed from the viewport toolbar (P8.2c).
/// Grid snap quantizes a translate to `grid_size` world units; **pixel snap**
/// (finer) quantizes to `1/pixels_per_unit` world units. Pixel snap takes
/// precedence when both are on. Rotate/scale keep their Shift-drag defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Snap2DSettings {
    pub grid_enabled: bool,
    pub grid_size: f32,
    pub pixel_enabled: bool,
    /// Per-project pixels-per-unit (default 100); a translate snaps to
    /// multiples of `1/ppu` world units when pixel snap is on.
    pub pixels_per_unit: f32,
}

impl Default for Snap2DSettings {
    fn default() -> Self {
        Self {
            grid_enabled: false,
            grid_size: 1.0,
            pixel_enabled: false,
            pixels_per_unit: 100.0,
        }
    }
}

impl Snap2DSettings {
    /// Translate snap increment in world units (`0.0` ⇒ no snap). Pixel snap
    /// wins over grid snap when both are enabled.
    pub fn translate_snap(&self) -> f32 {
        if self.pixel_enabled && self.pixels_per_unit > 0.0 {
            1.0 / self.pixels_per_unit
        } else if self.grid_enabled && self.grid_size > 0.0 {
            self.grid_size
        } else {
            0.0
        }
    }
}

/// The gizmo's orientation frame (Wave 2). `World` aligns the transform handles
/// to the world axes; `Local` aligns them to the primary selection's own
/// rotation. 2D mode forces `World` (the handles live in the sprite plane).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GizmoSpace {
    #[default]
    World,
    Local,
}

/// 3D transform-gizmo snap increments pushed from the toolbar (Wave 2). Replaces
/// the previously-hardcoded 1 m / 15° / 0.1 constants. When `always_on` is set,
/// every drag snaps; otherwise snapping is Shift-gated (holding Shift snaps,
/// preserving the pre-Wave-2 feel).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SnapSettings {
    /// Translate increment, world metres (`<= 0` ⇒ no translate snap).
    pub translate: f32,
    /// Rotate increment, degrees (`<= 0` ⇒ no rotate snap).
    pub rotate_deg: f32,
    /// Scale ratio increment (`<= 0` ⇒ no scale snap).
    pub scale: f32,
    /// Snap without holding Shift.
    pub always_on: bool,
}

impl Default for SnapSettings {
    fn default() -> Self {
        // The pre-Wave-2 Shift-drag defaults (1 m / 15° / 0.1 ratio), Shift-gated.
        Self {
            translate: 1.0,
            rotate_deg: 15.0,
            scale: 0.1,
            always_on: false,
        }
    }
}

/// Which viewport tool owns the left mouse button (P10.2b). `Select` is the
/// default pick + transform-gizmo interaction; `Sculpt` turns an LMB-drag over a
/// terrain entity into a height brush. Orthogonal to [`ViewportMode`] — sculpting
/// is a perspective-only tool (2D mode keeps `Select`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolMode {
    #[default]
    Select,
    Sculpt,
    /// Scatter foliage instances onto the terrain under an LMB-drag brush (E-P6,
    /// terrain-only placement v1). Perspective-only, like [`ToolMode::Sculpt`].
    Foliage,
    /// Paint per-sample **biome ids** onto the terrain under an LMB-drag brush
    /// (P19.2). Perspective-only, like [`ToolMode::Sculpt`].
    ///
    /// A tool mode of its own rather than a [`SculptOp`] sub-mode — the Foliage
    /// precedent, not the Paint one — because it edits a different layer with a
    /// different meaning: its "strength" moves a hard boundary rather than a blend
    /// (see `inf_terrain::biomepaint`), and it needs a *biome* picker where the
    /// sculpt controls need a layer picker. Folding it into [`SculptOp`] would
    /// have made [`SculptSettings::strength`] mean a third thing depending on the
    /// op.
    Biome,
    /// Place rivers and lakes against the terrain (P20.4). Perspective-only.
    ///
    /// Unlike the three brushes above this is **not** a drag-dab stroke: a river
    /// click *appends a control point*, and a lake drag *defines a rectangle*.
    /// The two sub-modes live in [`WaterSettings::kind`] rather than in two tool
    /// modes because they share the terrain pick, the "which body am I editing"
    /// state and the biome-hint defaults — the Sculpt/Paint precedent.
    Water,
    /// Carve (and fill) a voxel volume — caves, tunnels and excavations (P21.2).
    /// Perspective-only, like every other terrain tool.
    ///
    /// Its two sub-modes ([`VoxelToolKind`]) are a **brush** and a **spline
    /// tunnel**, and they live in [`VoxelSettings::kind`] for exactly the reason
    /// the water tool's do: they share the volume resolution, the surface-cut
    /// verdict, the carve/fill switch and the material — everything except how
    /// the author describes the path.
    Voxel,
}

/// Which cut the [`ToolMode::Voxel`] tool makes (P21.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VoxelToolKind {
    /// **Carve brush**: an LMB-drag lays sphere dabs along the stroke, spaced by
    /// arc length exactly as the sculpt brush's are, so drag speed cannot change
    /// what is dug.
    #[default]
    Brush,
    /// **Spline tunnel**: each click appends a waypoint, and Ctrl+click closes the
    /// path and tube-carves it as one capsule chain — one undo step for the whole
    /// tunnel.
    Tunnel,
    /// **Box cut** (P21.3): press-drag a rectangle on the surface and release to
    /// excavate it to [`VoxelSettings::depth_m`] below grade — the foundation
    /// pit, the parking garage, the underground mall.
    ///
    /// A drag rather than two clicks, because the shape an author is describing
    /// is a *rectangle* and a rubber-banded rectangle is how every tool in the
    /// world describes one. The lake tool's gesture exactly.
    BoxCut,
    /// **Trench cut** (P21.3): click waypoints, Ctrl+click to commit, and the
    /// path becomes a swept **rectangular** cut — the utility trench, the road
    /// cut, the foundation footing.
    ///
    /// The tunnel's gesture with a different section, and deliberately a
    /// separate sub-mode rather than a "square tunnel" flag: a tunnel is a bore
    /// through rock at depth and a trench is an open cut from the surface, so
    /// their defaults, their previews and their readouts differ even though the
    /// clicks are the same.
    Trench,
}

/// Where a [`ToolMode::Voxel`] dig puts the material it removes (P21.3).
///
/// Mirrors `inf_editor_core::scene::undo::SpoilChoice`, which is the type the
/// transaction actually takes; this is the toolbar's half of it, and the
/// difference between the two is exactly the picked site (which lives on the
/// host, not in the settings, because the author picks it in the viewport).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpoilMode {
    /// The material is removed from the world. Right for a cave — nobody
    /// barrows the spoil out of a tunnel in a level editor — and the P21.2
    /// behaviour.
    #[default]
    Off,
    /// Pile it at the deterministic default site: east of the cut, clear of its
    /// rim, standing on the ground there.
    Auto,
    /// Pile it where the author picked
    /// ([`VoxelSettings::pick_spoil_site`]). Falls back to `Auto` until a site
    /// has been picked, and says so on the readout rather than silently digging
    /// with a different rule than the toolbar shows.
    Site,
}

impl SpoilMode {
    /// Resolve the toolbar's mode plus the marker the author may have placed
    /// into the transaction's own [`SpoilChoice`].
    ///
    /// # Why this lives here rather than in `host.rs` (the M11 argument, again)
    ///
    /// This module is **not** `#[cfg]`-gated — only `host` is — so a rule
    /// written here is compiled and tested on every CI leg, including the Linux
    /// one. "Where does the soil go" decides committed geometry, which is
    /// exactly the class of rule the M11 ledger item says must not be invisible
    /// to a whole platform. The host keeps the *state* (the picked marker lives
    /// on the viewport thread, because the author places it with a click) and
    /// calls this for the *decision*.
    ///
    /// A `Site` mode with no marker resolves to `Auto` rather than refusing: the
    /// author asked for a heap, and standing it in the documented default place
    /// is a better answer than not digging. The readout says so.
    pub fn choice(self, site: Option<DVec3>) -> inf_editor_core::scene::undo::SpoilChoice {
        use inf_editor_core::scene::undo::SpoilChoice;
        match self {
            SpoilMode::Off => SpoilChoice::Discard,
            SpoilMode::Auto => SpoilChoice::Auto,
            SpoilMode::Site => match site {
                Some(p) if p.is_finite() => SpoilChoice::At(p),
                _ => SpoilChoice::Auto,
            },
        }
    }
}

/// Carve or fill — which way a [`ToolMode::Voxel`] cut runs (P21.2).
///
/// Not a `bool`: the ops it maps onto are named
/// [`Carve`](inf_voxel::VoxelOpKind::Carve) and
/// [`Fill`](inf_voxel::VoxelOpKind::Fill), and a `carve: bool` at three layers of
/// IPC is three chances to invert it silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VoxelOpMode {
    /// Remove material (CSG difference) — and open the heightfield above it.
    #[default]
    Carve,
    /// Add material (CSG union) — and close the heightfield above it.
    Fill,
}

/// Voxel-tool configuration pushed from the viewport toolbar (P21.2).
///
/// SI throughout (architecture rule 6): `radius_m` is world **metres**, and it is
/// the *cut* radius — the sphere of a brush dab, the tube radius of a tunnel — so
/// one slider means one thing in both sub-modes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VoxelSettings {
    /// Brush or spline tunnel.
    pub kind: VoxelToolKind,
    /// Cut radius, world metres.
    pub radius_m: f64,
    /// How far **below the picked surface** the cut's centre sits, world metres
    /// (clamped ≥ 0).
    ///
    /// This is what turns a surface pick into a *tunnel*, and it is why the tool
    /// needs no second click to say "how deep". At `0` the cut breaks the ground
    /// where the author points — a cave mouth. Past the radius it hollows rock
    /// with no mouth at all, which the surface-crossing verdict then allows on
    /// any terrain, inline included.
    ///
    /// A depth and not a free 3D point on purpose: the pick is the heightfield
    /// under the cursor, so the cut stays a function of *where the author aimed*
    /// rather than of how far away the camera happened to be. Same ruling the
    /// water tool's pick got, and it matters more here because a carve commits
    /// geometry.
    pub depth_m: f64,
    /// Carve or fill.
    pub mode: VoxelOpMode,
    /// The splat index a **fill** paints. Ignored by a carve, which normalizes
    /// emptied samples back to the default material (an empty voxel carries no
    /// material — see `inf_voxel::ops`).
    ///
    /// A voxel material index **is** a terrain splat index, which is what makes a
    /// cave wall shade like the hillside it opens out of; `project_voxel` reads
    /// the palette off the `Terrain` on the same entity.
    pub material: u8,
    /// **Dig to grade** (P21.3): a brush dab becomes a *column* from `depth_m`
    /// below the surface up to daylight, instead of a ball centred at depth.
    ///
    /// The one setting that turns the P21.2 carve brush into a freehand
    /// excavation brush: every dab reaches the sky, so a stroke leaves an open
    /// trench rather than a string of buried bubbles. Ignored by the other three
    /// sub-modes, which are open to the sky by construction.
    pub dig_to_depth: bool,
    /// Where the excavated material goes.
    pub spoil: SpoilMode,
    /// While `true`, a viewport click **moves the spoil site** instead of
    /// digging.
    ///
    /// A sticky mode rather than a one-shot arm, deliberately: the toolbar
    /// button stays lit, every click drags the marker somewhere else, and the
    /// author turns it off when the heap is where they want it. A one-shot would
    /// have to disarm itself on the host and re-sync a flag the toolbar owns,
    /// which is a desync waiting to happen (the `c4bd663` armed-hint lesson).
    pub pick_spoil_site: bool,
}

impl Default for VoxelSettings {
    fn default() -> Self {
        Self {
            kind: VoxelToolKind::Brush,
            // 2 m: four voxels across at the `VoxelVolume::voxel_size_m` default
            // of 0.5 m, so the first dab reads as a tunnel mouth rather than as a
            // pinprick or as half the hillside.
            radius_m: 2.0,
            // Zero, so the very first click an author makes breaks the ground and
            // they can SEE the tool worked. A cave that opens nowhere is the
            // harder default to debug.
            depth_m: 0.0,
            mode: VoxelOpMode::Carve,
            material: 0,
            // Off, so the carve brush behaves exactly as it did in P21.2 until
            // an author asks for an excavation. A cave that filled the hillside
            // beside it with spoil on the first click would be a surprise, and
            // the whole point of the default is that nothing surprising happens.
            dig_to_depth: false,
            spoil: SpoilMode::Off,
            pick_spoil_site: false,
        }
    }
}

/// Which body the water tool places (P20.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WaterToolKind {
    /// Click to append a control point to the active river; the first click with
    /// no river selected starts one.
    #[default]
    River,
    /// Drag a rectangle; the level comes from the ground under the first corner
    /// (or the biome's `water_hint`).
    Lake,
}

/// Water-tool configuration pushed from the viewport toolbar (P20.4).
///
/// SI throughout (architecture rule 6): metres and m/s. Every field is a value
/// for a **new** body — editing an existing one goes through Details or the
/// tool's own profile command, not through the brush push, so switching the tool
/// off and on cannot silently rewrite a river you already drew.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WaterSettings {
    /// River or lake.
    pub kind: WaterToolKind,
    /// Full width of a new river, metres.
    pub width_m: f64,
    /// Depth to the bed of a new river, metres.
    pub depth_m: f64,
    /// Surface flow speed of a new river, m/s. Negative reverses it.
    pub flow_m_s: f64,
    /// Added to the suggested still-water level, metres — so "a lake 2 m above
    /// the ground I clicked" needs no arithmetic from the author.
    ///
    /// There is deliberately **no level field** beside it: the level comes from
    /// where the author clicks — the painted biome's `water_hint` when it has
    /// one, otherwise the ground — resolved *per click* from the id-indexed hint
    /// table `EngineHost::set_water_hints` receives. A toolbar-supplied level (or
    /// a single pre-resolved hint) would be a number the author has to re-derive
    /// as they move across the terrain, and the P20.4 audit found exactly that
    /// field here, documented as pushed and passed `None` unconditionally.
    pub level_offset_m: f64,
}

impl Default for WaterSettings {
    fn default() -> Self {
        Self {
            kind: WaterToolKind::River,
            // The `WaterBody` component's own river defaults, so a river drawn
            // with the tool and one added through Add Component start identical.
            width_m: 8.0,
            depth_m: 1.5,
            flow_m_s: 1.5,
            level_offset_m: 0.0,
        }
    }
}

/// Biome-brush configuration pushed from the viewport toolbar (P19.2). The host
/// reads it when starting a stroke.
///
/// `radius` is world metres. `strength` is **not** a blend fraction: a biome id
/// is categorical, so the brush writes a crisp edge and `strength` decides where
/// that edge falls — see `inf_terrain::biomepaint::biome_claims`, which is the
/// one place the rule lives. `biome` is the id painted; `0`
/// (`inf_terrain::UNASSIGNED_BIOME`) is the eraser.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiomeSettings {
    /// Brush radius, world metres.
    pub radius: f64,
    /// Where the hard boundary falls, `[0, 1]` — `1` stamps the whole disk.
    pub strength: f64,
    /// The falloff curve the boundary contour is taken from.
    pub falloff: SculptFalloff,
    /// The biome id to write. `0` erases back to *unassigned*.
    pub biome: u8,
}

impl Default for BiomeSettings {
    fn default() -> Self {
        Self {
            radius: 8.0,
            // Full strength by default: a biome stamp is what an author reaches
            // for first, and a partial-strength default would look like a broken
            // brush rather than a smaller one.
            strength: 1.0,
            falloff: SculptFalloff::Smooth,
            biome: inf_terrain::UNASSIGNED_BIOME,
        }
    }
}

/// Foliage-brush configuration pushed from the viewport toolbar (E-P6). The host
/// reads it when starting/continuing a scatter stroke. `radius` is world metres,
/// `density` is target instances per m² of brush area, `kind` selects the palette
/// slot instances draw from, `scale_jitter` is the ± fractional scale spread, and
/// `seed` makes a stroke's scatter deterministically reproducible (with the
/// stroke index) — no wall-clock / thread-rng, per the determinism law.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FoliageSettings {
    /// Brush radius, world metres.
    pub radius: f64,
    /// Target instances per m² of brush area (before min-spacing rejection).
    pub density: f64,
    /// Erase mode: an LMB-drag removes instances within the radius instead of
    /// placing them. (Alt is reserved for the Alt-orbit gesture, so erase is a
    /// toolbar toggle in v1, not an Alt modifier.)
    pub erase: bool,
    /// Palette index (into `Foliage::palette`) new instances reference.
    pub kind: u32,
    /// Uniform-scale jitter: each instance scales `1 ± scale_jitter` (clamped ≥ 0).
    pub scale_jitter: f64,
    /// Align new instances to the terrain normal (v1: false — yaw-only; normal
    /// alignment is a documented follow-up).
    pub align_to_normal: bool,
    /// Deterministic scatter seed (folded with the per-stroke index + sample
    /// index through xxh3).
    pub seed: u32,
}

impl Default for FoliageSettings {
    fn default() -> Self {
        Self {
            radius: 3.0,
            density: 0.4,
            erase: false,
            kind: 0,
            scale_jitter: 0.2,
            align_to_normal: false,
            seed: 1,
        }
    }
}

/// The sculpt brush operation, mirrored from the toolbar. A flat, transport-
/// friendly enum the host maps onto `inf_terrain::BrushOp` (filling in the
/// op-specific parameters — smooth iterations, noise field, flatten target).
/// `Paint` is the P10.4 splat sub-mode: it edits per-sample layer **weights**
/// (via `inf_terrain::SplatStroke`) rather than heights, targeting
/// [`SculptSettings::paint_layer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SculptOp {
    #[default]
    Raise,
    Lower,
    Smooth,
    Flatten,
    Noise,
    /// Paint splat weight toward [`SculptSettings::paint_layer`] (P10.4).
    Paint,
}

/// Brush falloff curve, mirrored from the toolbar onto `inf_terrain::Falloff`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SculptFalloff {
    #[default]
    Smooth,
    Linear,
    Sphere,
    Sharp,
}

/// Sculpt brush configuration pushed from the viewport toolbar (P10.2b). The
/// host reads it when starting a stroke; `radius` is world metres, `strength` is
/// per-dab metres at full weight for Raise/Lower/Noise or a `[0,1]` blend
/// fraction for Smooth/Flatten.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SculptSettings {
    pub op: SculptOp,
    pub radius: f64,
    pub strength: f64,
    pub falloff: SculptFalloff,
    /// Target splat layer `0..=3` for the [`SculptOp::Paint`] sub-mode (P10.4).
    pub paint_layer: u8,
}

impl Default for SculptSettings {
    fn default() -> Self {
        Self {
            op: SculptOp::Raise,
            radius: 8.0,
            strength: 0.5,
            falloff: SculptFalloff::Smooth,
            paint_layer: 0,
        }
    }
}

/// The largest per-dab displacement the brush will accept, in metres at full
/// weight.
///
/// A thousand kilometres in one dab is already absurd — the point of the number
/// is not taste, it is that `f32::MAX` is `3.4 × 10³⁸` and the height a dab
/// writes goes through `as f32`, which **saturates rather than wrapping**. A
/// single unbounded dab could therefore make a tile infinite, and it takes
/// `10³²` bounded ones to do the same.
pub const MAX_SCULPT_STRENGTH: f64 = 1.0e6;

/// The largest brush radius, in world metres. Generous against any real terrain
/// and finite, which is the property that matters: `Falloff::weight` divides by
/// it.
pub const MAX_SCULPT_RADIUS: f64 = 1.0e6;

impl SculptSettings {
    /// Bound the brush's two continuous parameters — **the door every producer
    /// of a `SculptSettings` goes through** (C4-35).
    ///
    /// The Tauri command used to write `radius: sculpt.radius.max(0.0)` and, on
    /// the very next line, `strength: sculpt.strength` with nothing at all. Half
    /// a guard reads as a whole one, which is why the asymmetry survived: the
    /// line above it looks like the rule being followed.
    ///
    /// What the missing half cost: `Raise` computes `old + strength · w` in f64
    /// and narrows with `as f32`, which **saturates to `f32::INFINITY`** past
    /// `f32::MAX` instead of wrapping. A later `Smooth` dab over the same tile
    /// computes `old + (mean − old) · blend` — `inf − inf` — which is **NaN**,
    /// and `Flatten` does the same. Every guard on the way was NaN-blind, so the
    /// whole footprint was committed into the `.inf_terrain`.
    ///
    /// A non-finite value is replaced by the default rather than clamped: NaN
    /// has no nearest bound, and `clamp` would propagate it.
    pub fn sanitized(mut self) -> Self {
        let d = Self::default();
        self.radius = if self.radius.is_finite() {
            self.radius.clamp(0.0, MAX_SCULPT_RADIUS)
        } else {
            d.radius
        };
        self.strength = if self.strength.is_finite() {
            self.strength
                .clamp(-MAX_SCULPT_STRENGTH, MAX_SCULPT_STRENGTH)
        } else {
            d.strength
        };
        self.paint_layer = self.paint_layer.min(3);
        self
    }
}

/// Shortest-path angular lerp (handles the ±π wrap).
fn lerp_angle(a: f32, b: f32, t: f32) -> f32 {
    let mut d = (b - a) % std::f32::consts::TAU;
    if d > std::f32::consts::PI {
        d -= std::f32::consts::TAU;
    } else if d < -std::f32::consts::PI {
        d += std::f32::consts::TAU;
    }
    a + d * t
}

#[cfg(test)]
mod tests {
    /// **The spoil rule, tested where every CI leg can see it** (the M11 move,
    /// completed in the P21.3 audit round). It used to be `EngineHost::spoil_choice`,
    /// inside the `#[cfg(any(windows, macos))]` module.
    #[test]
    fn the_spoil_mode_resolves_the_marker_the_way_the_toolbar_shows() {
        use inf_editor_core::scene::undo::SpoilChoice;
        let here = DVec3::new(10.0, 2.0, -4.0);

        // Off discards, whatever marker is lying around.
        assert_eq!(SpoilMode::Off.choice(None), SpoilChoice::Discard);
        assert_eq!(SpoilMode::Off.choice(Some(here)), SpoilChoice::Discard);
        // Auto ignores the marker too — the toolbar says "east of the cut".
        assert_eq!(SpoilMode::Auto.choice(Some(here)), SpoilChoice::Auto);
        // Site uses it …
        assert_eq!(SpoilMode::Site.choice(Some(here)), SpoilChoice::At(here));
        // … falls back to the documented default when none has been picked …
        assert_eq!(SpoilMode::Site.choice(None), SpoilChoice::Auto);
        // … and never places a heap at infinity from a broken pick.
        assert_eq!(
            SpoilMode::Site.choice(Some(DVec3::splat(f64::NAN))),
            SpoilChoice::Auto
        );
        assert_eq!(
            SpoilMode::Site.choice(Some(DVec3::splat(f64::INFINITY))),
            SpoilChoice::Auto
        );
    }

    use super::*;

    #[test]
    fn pitch_clamps_at_poles() {
        let mut cam = EditorCamera::default();
        cam.apply_fly(
            &FlyInput {
                mouse_dy: -1e6,
                ..Default::default()
            },
            0.016,
        );
        assert!(cam.pitch <= PITCH_LIMIT);
        cam.apply_fly(
            &FlyInput {
                mouse_dy: 1e6,
                ..Default::default()
            },
            0.016,
        );
        assert!(cam.pitch >= -PITCH_LIMIT);
    }

    #[test]
    fn forward_moves_along_view_direction() {
        let mut cam = EditorCamera {
            pos: DVec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            fly_speed: 10.0,
        };
        cam.apply_fly(
            &FlyInput {
                forward: true,
                ..Default::default()
            },
            0.5,
        );
        assert!((cam.pos - DVec3::new(0.0, 0.0, -5.0)).length() < 1e-6);
    }

    #[test]
    fn wheel_scales_speed_with_clamp() {
        let mut cam = EditorCamera::default();
        cam.apply_fly(
            &FlyInput {
                wheel_steps: 100,
                ..Default::default()
            },
            0.016,
        );
        assert_eq!(cam.fly_speed, FLY_SPEED_MAX);
        cam.apply_fly(
            &FlyInput {
                wheel_steps: -200,
                ..Default::default()
            },
            0.016,
        );
        assert_eq!(cam.fly_speed, FLY_SPEED_MIN);
    }

    #[test]
    fn boost_quadruples_step() {
        let base = {
            let mut cam = EditorCamera::default();
            let p0 = cam.pos;
            cam.apply_fly(
                &FlyInput {
                    forward: true,
                    ..Default::default()
                },
                0.1,
            );
            (cam.pos - p0).length()
        };
        let boosted = {
            let mut cam = EditorCamera::default();
            let p0 = cam.pos;
            cam.apply_fly(
                &FlyInput {
                    forward: true,
                    boost: true,
                    ..Default::default()
                },
                0.1,
            );
            (cam.pos - p0).length()
        };
        assert!((boosted / base - 4.0).abs() < 1e-4);
    }

    #[test]
    fn diagonal_movement_is_normalized() {
        let mut cam = EditorCamera {
            pos: DVec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            fly_speed: 10.0,
        };
        cam.apply_fly(
            &FlyInput {
                forward: true,
                right: true,
                ..Default::default()
            },
            1.0,
        );
        assert!((cam.pos.length() - 10.0).abs() < 1e-6);
    }

    #[test]
    fn orbit_preserves_radius_and_keeps_looking_at_pivot() {
        let mut cam = EditorCamera {
            pos: DVec3::new(0.0, 0.0, 10.0),
            yaw: 0.0,
            pitch: 0.0,
            fly_speed: 8.0,
        };
        let pivot = DVec3::ZERO;
        let r0 = (cam.pos - pivot).length();
        cam.apply_navigate(
            &NavInput {
                mode: NavMode::Orbit,
                mouse_dx: 40.0,
                mouse_dy: 15.0,
                ..Default::default()
            },
            pivot,
            0.016,
        );
        let r1 = (cam.pos - pivot).length();
        assert!((r1 - r0).abs() < 1e-4, "radius drifted {r0} -> {r1}");
        // Still aimed at the pivot.
        let dir = (pivot - cam.pos).as_vec3().normalize();
        assert!(dir.dot(cam.forward()) > 0.999);
    }

    #[test]
    fn dolly_moves_toward_pivot_without_crossing() {
        let mut cam = EditorCamera {
            pos: DVec3::new(0.0, 0.0, 10.0),
            yaw: 0.0,
            pitch: 0.0,
            fly_speed: 8.0,
        };
        let pivot = DVec3::ZERO;
        cam.dolly(0.5, pivot); // pull halfway in
        assert!((cam.pos.z - 5.0).abs() < 1e-4, "pos {:?}", cam.pos);
        // A huge dolly clamps just short of the pivot, never past it.
        cam.dolly(100.0, pivot);
        assert!(cam.pos.z > 0.0 && cam.pos.z < 0.1);
    }

    #[test]
    fn pan_scales_with_distance() {
        let mut near = EditorCamera {
            pos: DVec3::new(0.0, 0.0, 2.0),
            yaw: 0.0,
            pitch: 0.0,
            fly_speed: 8.0,
        };
        let mut far = EditorCamera {
            pos: DVec3::new(0.0, 0.0, 40.0),
            yaw: 0.0,
            pitch: 0.0,
            fly_speed: 8.0,
        };
        let input = NavInput {
            mode: NavMode::Pan,
            mouse_dx: 50.0,
            ..Default::default()
        };
        near.apply_navigate(&input, DVec3::ZERO, 0.016);
        far.apply_navigate(&input, DVec3::ZERO, 0.016);
        // Farther pivot → larger pan for the same drag.
        assert!(far.pos.x.abs() > near.pos.x.abs() * 5.0);
    }

    #[test]
    fn focus_goal_frames_target_and_advance_settles() {
        let mut cam = EditorCamera {
            pos: DVec3::new(0.0, 0.0, 100.0),
            yaw: 0.0,
            pitch: 0.0,
            fly_speed: 8.0,
        };
        let target = DVec3::new(5.0, 1.0, -3.0);
        let goal = cam.focus_goal(target, 2.0);
        // Goal keeps the target ahead and within a reasonable distance.
        let dist = (goal.pos - target).length();
        assert!((3.0..12.0).contains(&dist), "focus dist {dist}");
        // Advancing many frames converges exactly.
        let mut settled = false;
        for _ in 0..600 {
            if cam.advance_focus(&goal, 0.016) {
                settled = true;
                break;
            }
        }
        assert!(settled);
        assert_eq!(cam.pos, goal.pos);
    }

    #[test]
    fn bookmarks_store_and_recall() {
        let mut bm = Bookmarks::default();
        let cam = EditorCamera::default();
        assert!(bm.recall(3).is_none());
        bm.store(3, cam.pose());
        assert_eq!(bm.recall(3).unwrap(), cam.pose());
        // Out-of-range slots are ignored, not panicking.
        bm.store(0, cam.pose());
        bm.store(99, cam.pose());
        assert!(bm.recall(0).is_none());
        assert!(bm.recall(99).is_none());
    }

    #[test]
    fn two_d_zoom_keeps_world_point_under_cursor() {
        // The world point beneath the cursor must not move as we zoom in/out.
        let (w, h) = (1600.0, 900.0);
        let (px, py) = (1180.0, 300.0); // an off-center cursor
        let mut cam = Camera2D {
            center: DVec2::new(3.0, -2.0),
            half_height: 8.0,
        };
        let before = cam.world_at_pixel(px, py, w, h);
        cam.zoom_at(3, px, py, w, h); // zoom in three detents
        let after = cam.world_at_pixel(px, py, w, h);
        assert!(
            (before - after).length() < 1e-9,
            "zoom-to-cursor drifted: {before:?} -> {after:?}"
        );
        assert!(cam.half_height < 8.0, "scroll up should zoom in");
        // And out again returns the invariant too.
        cam.zoom_at(-5, px, py, w, h);
        let out = cam.world_at_pixel(px, py, w, h);
        assert!((before - out).length() < 1e-9, "zoom-out drifted");
    }

    #[test]
    fn two_d_zoom_clamps_half_height() {
        let (w, h) = (800.0, 600.0);
        let mut cam = Camera2D::default();
        cam.zoom_at(1000, 400.0, 300.0, w, h);
        assert!(cam.half_height >= TWO_D_HALF_HEIGHT_MIN - 1e-9);
        cam.zoom_at(-1000, 400.0, 300.0, w, h);
        assert!(cam.half_height <= TWO_D_HALF_HEIGHT_MAX + 1e-9);
    }

    #[test]
    fn two_d_pan_tracks_the_grabbed_point() {
        // Panning by a pixel delta shifts the world under the cursor by exactly
        // that many world units (so the grabbed point stays put visually).
        let (w, h) = (1000.0, 500.0);
        let mut cam = Camera2D {
            center: DVec2::ZERO,
            half_height: 5.0,
        };
        let p0 = cam.world_at_pixel(500.0, 250.0, w, h);
        cam.pan(40.0, -20.0, w, h);
        // The grabbed world point tracks the cursor: after panning by (dx, dy)
        // it sits under the pixel that moved by (dx, dy).
        let p1 = cam.world_at_pixel(500.0 + 40.0, 250.0 - 20.0, w, h);
        assert!((p0 - p1).length() < 1e-9, "pan didn't track: {p0:?} {p1:?}");
    }

    #[test]
    fn mode_switch_preserves_each_camera_pose() {
        // The platform loop keeps a perspective `EditorCamera` and a `Camera2D`
        // side by side and drives only the active one, so switching modes
        // restores the exact pose. Model that independence directly.
        let mut cam = EditorCamera::default();
        let persp_pose = cam.pose();
        let mut cam2d = Camera2D::default();

        // "2D mode": pan + zoom the 2D camera — the flycam pose is untouched.
        cam2d.pan(30.0, 10.0, 1600.0, 900.0);
        cam2d.zoom_at(2, 800.0, 450.0, 1600.0, 900.0);
        assert_eq!(cam.pose(), persp_pose, "flycam drifted during 2D nav");
        let two_d_state = (cam2d.center, cam2d.half_height);

        // "Perspective mode": fly the flycam — the 2D camera keeps its exact
        // panned/zoomed state, so returning to 2D restores it.
        cam.apply_fly(
            &FlyInput {
                forward: true,
                ..Default::default()
            },
            0.2,
        );
        assert_ne!(cam.pose(), persp_pose, "flycam should have moved");
        assert_eq!(
            (cam2d.center, cam2d.half_height),
            two_d_state,
            "2D camera drifted during perspective nav"
        );
    }

    #[test]
    fn snap_2d_pixel_precedence_and_increments() {
        // Default: no snap.
        assert_eq!(Snap2DSettings::default().translate_snap(), 0.0);
        // Grid only: snap to grid_size.
        let grid = Snap2DSettings {
            grid_enabled: true,
            grid_size: 0.25,
            ..Default::default()
        };
        assert_eq!(grid.translate_snap(), 0.25);
        // Pixel snap: 1/ppu, and it wins over grid.
        let pix = Snap2DSettings {
            grid_enabled: true,
            grid_size: 0.25,
            pixel_enabled: true,
            pixels_per_unit: 100.0,
        };
        assert!((pix.translate_snap() - 0.01).abs() < 1e-9);
        // Zero/negative ppu degrades to no pixel snap (falls back to grid).
        let bad = Snap2DSettings {
            grid_enabled: true,
            grid_size: 0.5,
            pixel_enabled: true,
            pixels_per_unit: 0.0,
        };
        assert_eq!(bad.translate_snap(), 0.5);
    }

    #[test]
    fn two_d_frame_centers_and_fits() {
        let mut cam = Camera2D::default();
        cam.frame(DVec2::new(10.0, 5.0), DVec2::new(3.0, 2.0), 16.0 / 9.0);
        assert_eq!(cam.center, DVec2::new(10.0, 5.0));
        // Half-height must cover the taller of (y extent, x extent / aspect).
        assert!(cam.half_height >= 2.0);
        assert!(cam.half_height * (16.0 / 9.0) >= 3.0);
    }

    #[test]
    fn lerp_angle_takes_short_way_around_wrap() {
        // From +170° to -170° is +20° across the seam, not -340°.
        let a = 170f32.to_radians();
        let b = (-170f32).to_radians();
        let mid = lerp_angle(a, b, 0.5);
        // Halfway should be at ±180°, not near 0.
        assert!(mid.abs() > 175f32.to_radians());
    }
}
