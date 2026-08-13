//! **Virtual-shadow-map geometry** (P27.1): the per-light projections, the
//! clipmap's centring rule, and the one level rule the marking pass mirrors.
//!
//! The `csm.rs` analogue for virtual shadows — every piece of arithmetic here is
//! a pure function so it unit-tests without a device, and `inf-vsm` holds the
//! address space these matrices index into.
//!
//! # The depth convention: pages are **reverse-Z**, like the camera
//!
//! `docs/memos/p27-1-depth-convention.md` carries the measurement; the ruling
//! and its two numbers live here, beside the constants they fix.
//!
//! `csm.rs` documents the CSM exception — *"unlike the reverse-Z camera
//! projection, the shadow ortho uses a conventional forward-Z range"* — on the
//! grounds that it "simplifies the bias reasoning". P27 was asked to keep that
//! exception only if measurement defended it. It does not:
//!
//! * **Orthographic (the directional clipmap) does not distinguish them.** A
//!   clipmap's depth is linear in view distance, so one f32 step is
//!   `ulp(d) × range` whichever way round it runs. Measured over a 200 m box:
//!   the worst case is **21.2 µm** for reverse-Z and **15.2 µm** for forward-Z —
//!   a ratio of **1.40**, and both four orders of magnitude under any bias this
//!   engine uses (the CSM's own `depth_bias` default of 0.0015 NDC *is* 0.3 m
//!   over that range).
//!
//!   The intuition that forward-Z is coarse "at the far plane" and reverse-Z
//!   "at the near plane" is **wrong**, and this measurement is what caught it:
//!   f32's ULP is a step function of the *stored* depth, so a linear depth's
//!   coarsest step lands wherever the `[0.5, 1)` binade does — at 67 m and
//!   153 m of a 200 m box respectively, not at either end.
//! * **Perspective (spot and point — which P27 adds and CSM never had) is not a
//!   tie.** Over a 0.1 m → 50 m spot frustum the worst-case depth resolution is
//!   **1.87 mm** under forward-Z (at 46.8 m, the far end of the cone, where the
//!   intuition does hold because a perspective depth is not linear) and
//!   **7.43 µm** under reverse-infinite Z: a factor of **251**. Forward-Z spends
//!   its precision where nothing is and starves the far half of the cone, which
//!   is exactly where a spot's shadow is largest on screen.
//! * And the exception costs something real that the CSM never had to pay:
//!   P27.2 rasterizes **every** light kind into **one** atlas, so two
//!   conventions would mean two depth-compare states, two clear values and two
//!   receiver comparisons keyed on light kind — in a pass that already has to
//!   scissor per page.
//!
//! So [`VSM_DEPTH_CLEAR`] and [`VSM_DEPTH_COMPARE`] are the camera's
//! ([`crate::camera::DEPTH_CLEAR`] / [`crate::camera::DEPTH_COMPARE`]), asserted
//! equal rather than repeated. The CSM path keeps its forward-Z exactly as it
//! is — P27.5 demotes that path, it does not edit it.

use glam::{Mat4, Vec3};

use crate::camera::{ortho_reverse_z, RenderView, DEPTH_CLEAR, DEPTH_COMPARE};
use crate::scene::{LightKind, RenderScene};
use crate::settings::VsmSettings;
use inf_vsm::{VsmLightDesc, VsmPage, VsmTreeKind, VSM_PAGE_SIZE};

/// Shadow pages clear to the camera's clear value. See the module docs.
pub const VSM_DEPTH_CLEAR: f32 = DEPTH_CLEAR;

/// Shadow pages compare with the camera's comparison. See the module docs.
pub const VSM_DEPTH_COMPARE: wgpu::CompareFunction = DEPTH_COMPARE;

/// The most (light × face) projections one frame may mark through.
///
/// A VRAM number rather than a quality one — 64 × 96 B is 6 KiB — and the
/// degradation past it is the honest one: the tail of the light list marks
/// nothing, so those lights keep whatever pages they already had. Six of these
/// are one point light, so this is ten point lights or sixty-four suns.
pub const VSM_MAX_PROJECTIONS: usize = 64;

/// The six cube-face `(forward, up)` bases, in face order. A point light's
/// quadtree faces are indexed by this array, so the CPU twin and the marking
/// pass cannot disagree about which face a direction lands on.
pub const CUBE_FACE_BASES: [(Vec3, Vec3); 6] = [
    (Vec3::X, Vec3::Y),
    (Vec3::NEG_X, Vec3::Y),
    (Vec3::Y, Vec3::Z),
    (Vec3::NEG_Y, Vec3::NEG_Z),
    (Vec3::Z, Vec3::Y),
    (Vec3::NEG_Z, Vec3::Y),
];

/// The world size of one **level-0** page of a clipmap whose level-0 half-extent
/// is `half_extent` and whose levels are `pages_per_side` pages across.
#[inline]
pub fn clipmap_page_world(half_extent: f32, pages_per_side: u32) -> f32 {
    2.0 * half_extent / pages_per_side.max(1) as f32
}

/// **Where a clipmap is centred**: the camera's render-local eye, snapped to the
/// *level-0 page* lattice.
///
/// Snapping is `csm::cascade_matrix`'s texel snap one granularity up, and for
/// the same reason: an unsnapped centre makes every page's content depend on
/// sub-page camera motion, so a static scene would re-rasterize everything every
/// frame and P27.3's "zero page re-rasters after warm-up" would be unreachable.
///
/// **The honest bound, carried to P27.3.** All levels share this one centre,
/// which is what makes `inf_vsm`'s concentric `N/4 + x/2` parent rule exact. The
/// price is that a *coarse* level's grid moves in steps of a level-0 page —
/// a fraction of its own page — so a coarse page's content shifts on any camera
/// motion. That costs nothing in P27.1, where nothing is cached and every marked
/// page is re-allocated each frame; it is the first thing P27.3's caching clause
/// has to answer, and per-level offsets are the answer it will need.
pub fn clipmap_centre(eye: Vec3, page_world: f32) -> Vec3 {
    let p = page_world.max(1e-4);
    Vec3::new(
        (eye.x / p).round() * p,
        (eye.y / p).round() * p,
        (eye.z / p).round() * p,
    )
}

/// The **level-0** `view_proj` of a directional light's clipmap: a reverse-Z
/// orthographic box of `half_extent` about `centre`, looking along the light.
///
/// Level `L`'s projection is this one with the x/y extent scaled by `2^L` about
/// the same centre, which is a division of the NDC rather than a second matrix —
/// so one matrix per directional light reaches every level, and the marking pass
/// derives the rest. `depth_range` is the along-light extent the box spans, and
/// the eye is pulled back half of it so casters behind the centre still fit.
pub fn clipmap_matrix(
    light_dir_to: Vec3,
    centre: Vec3,
    half_extent: f32,
    depth_range: f32,
) -> Mat4 {
    let fwd = (-light_dir_to).normalize_or_zero();
    let fwd = if fwd.length_squared() < 1e-6 {
        Vec3::NEG_Y
    } else {
        fwd
    };
    let up = if fwd.dot(Vec3::Y).abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let right = fwd.cross(up).normalize_or_zero();
    let true_up = right.cross(fwd).normalize_or_zero();
    let range = depth_range.max(1e-3);
    let eye = centre - fwd * (range * 0.5);
    let view = glam::camera::rh::view::look_to_mat4(eye, fwd, true_up);
    // Reverse-Z, near → 1 and far → 0 — the camera's convention, through the
    // camera's own function. See the module docs for the measurement.
    ortho_reverse_z(half_extent.max(1e-3), 1.0, 1e-3, range) * view
}

/// The `view_proj` of a spot light's page tree: a reverse-infinite-Z perspective
/// down the beam, at the outer cone's full angle.
///
/// `outer_cos` is `RenderLight::outer_cos` — the cosine of the **half**-angle —
/// so the projection's vertical field of view is twice its arccos, clamped away
/// from the degenerate ends.
pub fn spot_matrix(position: Vec3, light_dir_to: Vec3, outer_cos: f32, near: f32) -> Mat4 {
    let fwd = (-light_dir_to).normalize_or_zero();
    let fwd = if fwd.length_squared() < 1e-6 {
        Vec3::NEG_Y
    } else {
        fwd
    };
    let up = if fwd.dot(Vec3::Y).abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let right = fwd.cross(up).normalize_or_zero();
    let true_up = right.cross(fwd).normalize_or_zero();
    let view = glam::camera::rh::view::look_to_mat4(position, fwd, true_up);
    glam::camera::rh::proj::directx::perspective_infinite_reverse(
        spot_fov_y(outer_cos),
        1.0,
        near.max(1e-3),
    ) * view
}

/// The vertical field of view a spot's outer cone needs, in radians.
///
/// Clamped into `[2°, 170°]`: an outer cosine of exactly 1 is a zero-width
/// projection matrix and one of −1 is a 360° one, and a light's cone is authored
/// content that can be either.
#[inline]
pub fn spot_fov_y(outer_cos: f32) -> f32 {
    (2.0 * outer_cos.clamp(-1.0, 1.0).acos()).clamp(2.0_f32.to_radians(), 170.0_f32.to_radians())
}

/// The `view_proj` of one face of a point light's cube: a 90° reverse-infinite-Z
/// perspective along [`CUBE_FACE_BASES`]`[face]`.
pub fn cube_face_matrix(position: Vec3, face: usize, near: f32) -> Mat4 {
    let (fwd, up) = CUBE_FACE_BASES[face.min(5)];
    let view = glam::camera::rh::view::look_to_mat4(position, fwd, up);
    glam::camera::rh::proj::directx::perspective_infinite_reverse(
        std::f32::consts::FRAC_PI_2,
        1.0,
        near.max(1e-3),
    ) * view
}

/// **Which level a receiver pixel justifies** — the ONE rule, on the CPU,
/// mirrored by `vsm_mark.wgsl`.
///
/// `texel0_world` is the world size of one shadow texel at level 0 *at this
/// point*, `pixel_world` the world size of one screen pixel there. A level whose
/// texels are `2^L` times coarser is the right one when its texels land about one
/// per pixel, so `L = ceil(log2(pixel_world / texel0_world))`, clamped into the
/// tree.
///
/// `ceil` and not `round`, and the direction is `inf_render::justified_mip`'s for
/// the same reason inverted: erring **coarse** costs a blurrier shadow, erring
/// fine costs pages that will be deferred and a want set that never converges.
pub fn vsm_justified_level(texel0_world: f32, pixel_world: f32, levels: u32) -> u32 {
    if levels == 0 {
        return 0;
    }
    let want = (pixel_world.max(1e-6) / texel0_world.max(1e-9))
        .log2()
        .ceil();
    want.clamp(0.0, (levels - 1) as f32) as u32
}

/// **The coarsest level a clipmap point is outside of** — the other half of the
/// clipmap's level rule.
///
/// `ndc0` is the point's level-0 NDC extent (`max(|x|, |y|)`). Level `L` covers
/// `2^L` times level 0, so the finest level that *contains* the point is
/// `ceil(log2(ndc0))`. `None` when even the coarsest level does not reach it,
/// which is the honest "outside the shadow range" answer: no page is marked, no
/// shadow is stored, and the receiver reads lit.
///
/// The marked level is the **max** of this and [`vsm_justified_level`]: a pixel
/// cannot be served by a level that does not cover it, however fine its footprint
/// says it deserves.
pub fn clipmap_containing_level(ndc0: f32, levels: u32) -> Option<u32> {
    if levels == 0 {
        return None;
    }
    if ndc0 <= 1.0 {
        return Some(0);
    }
    let l = ndc0.log2().ceil();
    if l > (levels - 1) as f32 {
        None
    } else {
        Some(l as u32)
    }
}

/// **The world→clip matrix of ONE page** — the projection P27.2's caster raster
/// draws that page's dirty depth with.
///
/// A page is a *sub-rectangle* of its level's projection, so this is the light's
/// own `view_proj` composed with a 2-D scale-and-offset that blows that rectangle
/// up to the full `[-1, 1]` clip square. Nothing else changes: `z` and `w` pass
/// through untouched, so a page keeps the light's depth range, its reverse-Z
/// convention and its near plane exactly.
///
/// # The two level rules, inverted
///
/// [`mark_page_for`] decides which page a world point marks; this decides which
/// world points a page contains, and the two must be the same statement or the
/// raster fills pages the marker never asked for. So the arithmetic is inverted
/// from that function rather than re-derived:
///
/// * a **clipmap** level covers `2^L` times level 0 about the same centre, so its
///   NDC is level 0's divided by `2^L` — the `xy = l.xy / exp2(level)` line in
///   `vsm_mark.wgsl`;
/// * a **quadtree** (spot) or **cube face** level covers the light's whole
///   frustum whatever its page count, so its NDC *is* the light's.
///
/// # The vertical flip lives here too
///
/// Page row 0 is the **top** of the level — `mark_page_for` computes
/// `py = (0.5 − ndc.y·0.5) · pages_y`, so `ndc.y = +1` is row 0 — and wgpu's
/// viewport puts `ndc.y = +1` at the top of the rect. The two agree, which is why
/// a page's own projection needs no flip of its own: flipping here as well would
/// mirror every page's content vertically inside a correctly-chosen slot, which is
/// the defect that looks like a bias problem.
pub fn vsm_page_matrix(
    light_vp: Mat4,
    kind: VsmTreeKind,
    level: u32,
    pages_x: u32,
    pages_y: u32,
    x: u32,
    y: u32,
) -> Mat4 {
    let (nx, ny) = (pages_x.max(1) as f32, pages_y.max(1) as f32);
    let level_scale = match kind {
        VsmTreeKind::Clipmap => 1.0 / (1u64 << level.min(62)) as f32,
        VsmTreeKind::Quadtree | VsmTreeKind::Cube => 1.0,
    };
    // The page's centre in its LEVEL's NDC. `x` runs with NDC x; `y` runs against
    // it (row 0 is the top), which is the one asymmetry in this file.
    let cx = -1.0 + (2.0 * x as f32 + 1.0) / nx;
    let cy = 1.0 - (2.0 * y as f32 + 1.0) / ny;
    let (a, b) = (level_scale * nx, level_scale * ny);
    // `sub * clip` = `(a·clip.x − cx·nx·clip.w, b·clip.y − cy·ny·clip.w, clip.z,
    // clip.w)`: the level scale and the page's blow-up folded into one matrix, so
    // the raster pays one multiply and the light's own matrix is never rebuilt.
    let sub = Mat4::from_cols(
        glam::Vec4::new(a, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, b, 0.0, 0.0),
        glam::Vec4::new(0.0, 0.0, 1.0, 0.0),
        glam::Vec4::new(-cx * nx, -cy * ny, 0.0, 1.0),
    );
    sub * light_vp
}

/// Whether a world-space sphere can contribute to the page `page_vp` describes —
/// **the CPU twin of `vsm_cull.wgsl`'s test**, and the terrain caster path's own
/// cull.
///
/// The six clip-space planes of the matrix, tested against the sphere. A plane
/// whose normal is degenerate is skipped rather than trusted: a reverse-**infinite**
/// perspective (every spot and every cube face) has no far plane at all, and its
/// `w − z` row is identically zero — dividing by that length would make every
/// caster of every perspective light fail the test, which is the shape of a bug
/// that deletes all point-light shadows and nothing else.
///
/// Conservative by construction: it may keep a sphere the page cannot see, never
/// drop one it can. That direction is the one a *subtractive* cull needs, and it
/// is the same direction `inf_render::passes::vgeom`'s cull errs in.
pub fn vsm_page_sees_sphere(page_vp: &Mat4, centre: Vec3, radius: f32) -> bool {
    let r = page_vp.row(3);
    let planes = [
        page_vp.row(0) + r, // x ≥ −w
        r - page_vp.row(0), // x ≤ +w
        page_vp.row(1) + r, // y ≥ −w
        r - page_vp.row(1), // y ≤ +w
        page_vp.row(2),     // z ≥ 0
        r - page_vp.row(2), // z ≤ w — degenerate under an infinite far plane
    ];
    for p in planes {
        let n = p.truncate();
        let len = n.length();
        if len < 1e-6 {
            continue;
        }
        if (n.dot(centre) + p.w) / len < -radius {
            return false;
        }
    }
    true
}

/// One (light × face) projection, as the marking shader reads it — 96 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VsmProjection {
    /// Render-local world → this light/face's **level-0** clip space.
    pub view_proj: [f32; 16],
    /// x = word offset of this light's block in the indirection table,
    /// y = the light's first bit in the mask, z = the face index,
    /// w = kind: 0 = orthographic clipmap, 1 = perspective (spot / cube face).
    pub info: [u32; 4],
    /// xyz = the light's render-local position (perspective only), w = the
    /// level-0 shadow texel size: **world metres** for a clipmap, **metres per
    /// metre of light distance** for a perspective light.
    pub light: [f32; 4],
}

/// Kind discriminants carried in [`VsmProjection::info`]`.w`. **Freeze-pinned**:
/// the shader branches on the number.
pub const VSM_PROJ_ORTHO: u32 = 0;
/// See [`VSM_PROJ_ORTHO`].
pub const VSM_PROJ_PERSPECTIVE: u32 = 1;

/// The uniform the marking pass reads — 96 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VsmMarkParams {
    /// Camera clip → render-local world, for the depth reconstruction.
    pub inv_view_proj: [f32; 16],
    /// xyz = render-local eye, w = pixels per world unit at one metre
    /// (`inf_render::projection_scale` — the same door the VT floor uses).
    pub eye: [f32; 4],
    /// x = projection count, y = mask words, z = viewport width, w = height.
    pub counts: [u32; 4],
}

/// **Which page tree each shadow-casting light in `scene` gets**, in scene order.
///
/// The list is capped at [`VSM_MAX_PROJECTIONS`] *projections*, not lights, so a
/// point light costs six of the budget — which is what it costs the marking pass.
///
/// # The cap **stops**, it does not skip
///
/// A light that does not fit ends the list, and every shadow-caster after it is
/// dropped too — even one that would have fitted. That is not tidiness, it is
/// the invariant the whole mapping rests on: **handle `n` is the `n`-th
/// shadow-casting light in scene order**, and [`vsm_projections`] re-walks
/// `scene.lights` to rebuild it.
///
/// Skipping instead would break it silently and expensively. Eleven point lights
/// followed by a sun would drop the eleventh point light (66 > 64 projections)
/// and keep the sun — which then becomes handle 10, takes the *point* light's
/// table block and its bit range, and marks the sun's clipmap pages into another
/// light's address space. No error, no counter, and a shadow atlas full of pages
/// nothing will ever read.
pub fn vsm_light_trees(scene: &RenderScene, settings: &VsmSettings) -> Vec<VsmLightDesc> {
    let mut out = Vec::new();
    let mut projections = 0usize;
    for l in &scene.lights {
        if !l.cast_shadows {
            continue;
        }
        let desc = match l.kind {
            LightKind::Directional => {
                VsmLightDesc::clipmap(settings.clipmap_levels, settings.clipmap_pages_per_side)
            }
            LightKind::Spot => VsmLightDesc::quadtree(settings.spot_levels),
            LightKind::Point => VsmLightDesc::cube(settings.point_levels),
        };
        let faces = desc.faces() as usize;
        if projections + faces > VSM_MAX_PROJECTIONS {
            break;
        }
        projections += faces;
        out.push(desc);
    }
    out
}

/// **Build this frame's projection list** — one entry per (light × face), in the
/// same order [`vsm_light_trees`] produced the light list.
///
/// `blocks` and `bases` are the table word offset and the first mask bit of each
/// light, in handle order: both are already known exactly (the residency's
/// `table_block` and the mark layout's `light_base`), and a shader that
/// re-derived either would be a second copy of the layout — `vt_stream`'s
/// argument, kept.
pub fn vsm_projections(
    scene: &RenderScene,
    view: &RenderView,
    settings: &VsmSettings,
    trees: &[VsmLightDesc],
    blocks: &[u32],
    bases: &[u32],
) -> Vec<VsmProjection> {
    let mut out = Vec::new();
    let eye = view.eye_local();
    let page0 = clipmap_page_world(
        settings.first_level_extent_m,
        settings.clipmap_pages_per_side,
    );
    let centre = clipmap_centre(eye, page0);
    let mut handle = 0usize;
    for l in &scene.lights {
        if !l.cast_shadows {
            continue;
        }
        let (Some(tree), Some(&block), Some(&base)) =
            (trees.get(handle), blocks.get(handle), bases.get(handle))
        else {
            break;
        };
        handle += 1;
        let texels0 = (tree.levels[0].pages_x * VSM_PAGE_SIZE).max(1) as f32;
        match tree.kind {
            VsmTreeKind::Clipmap => {
                let half = settings.first_level_extent_m.max(1e-3);
                // The along-light span: the coarsest level's diameter, so a caster
                // anywhere in the clipmap's footprint is inside the box whatever
                // direction the sun comes from.
                let range = 2.0 * half * (1u32 << tree.coarsest_level()) as f32;
                out.push(VsmProjection {
                    view_proj: clipmap_matrix(l.direction.normalize_or_zero(), centre, half, range)
                        .to_cols_array(),
                    info: [block, base, 0, VSM_PROJ_ORTHO],
                    light: [0.0, 0.0, 0.0, 2.0 * half / texels0],
                });
            }
            VsmTreeKind::Quadtree => {
                let pos = view.origin.to_render(l.position);
                let per_metre = 2.0 * (spot_fov_y(l.outer_cos) * 0.5).tan() / texels0;
                out.push(VsmProjection {
                    view_proj: spot_matrix(
                        pos,
                        l.direction.normalize_or_zero(),
                        l.outer_cos,
                        settings.perspective_near_m,
                    )
                    .to_cols_array(),
                    info: [block, base, 0, VSM_PROJ_PERSPECTIVE],
                    light: [pos.x, pos.y, pos.z, per_metre],
                });
            }
            VsmTreeKind::Cube => {
                let pos = view.origin.to_render(l.position);
                // A 90° face: the cross-section is `2 · d · tan(45°)` = `2 · d`.
                let per_metre = 2.0 / texels0;
                for face in 0..6u32 {
                    out.push(VsmProjection {
                        view_proj: cube_face_matrix(
                            pos,
                            face as usize,
                            settings.perspective_near_m,
                        )
                        .to_cols_array(),
                        info: [block, base, face, VSM_PROJ_PERSPECTIVE],
                        light: [pos.x, pos.y, pos.z, per_metre],
                    });
                }
            }
        }
    }
    out
}

/// **The CPU twin of the marking rule**: which page of `proj`'s light a world
/// point at `world` marks, or `None` when it marks nothing.
///
/// The shader's arithmetic, in Rust, so a test can say what the GPU should have
/// produced from the same inputs rather than assert that it produced *something*.
/// `pixel_world` is the world size of one screen pixel at that point.
pub fn mark_page_for(
    proj: &VsmProjection,
    desc: &VsmLightDesc,
    world: Vec3,
    pixel_world: f32,
) -> Option<VsmPage> {
    let vp = Mat4::from_cols_array(&proj.view_proj);
    let clip = vp * world.extend(1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    // Outside the light's own depth range: in front of its near plane, or (for
    // the clipmap's finite box) behind its far plane.
    if ndc.z <= 0.0 || ndc.z > 1.0 {
        return None;
    }
    let levels = desc.level_count();
    let ortho = proj.info[3] == VSM_PROJ_ORTHO;
    let texel0 = if ortho {
        proj.light[3]
    } else {
        let d = (world - Vec3::new(proj.light[0], proj.light[1], proj.light[2])).length();
        proj.light[3] * d.max(1e-4)
    };
    let mut level = vsm_justified_level(texel0, pixel_world, levels);
    let uv = if ortho {
        // A clipmap point must also be INSIDE the level it is served by.
        level = level.max(clipmap_containing_level(
            ndc.x.abs().max(ndc.y.abs()),
            levels,
        )?);
        ndc.truncate() / (1u32 << level) as f32
    } else {
        ndc.truncate()
    };
    if uv.x.abs() > 1.0 || uv.y.abs() > 1.0 {
        return None;
    }
    let g = desc.levels[level as usize];
    // NDC y is up and the page grid's y runs down (the table's `(y, x)` order),
    // so the vertical axis flips here and in the shader — one convention, stated
    // where both of them can see it.
    let px = ((uv.x * 0.5 + 0.5) * g.pages_x as f32) as u32;
    let py = ((0.5 - uv.y * 0.5) * g.pages_y as f32) as u32;
    Some(VsmPage::new(
        proj.info[2],
        level,
        px.min(g.pages_x - 1),
        py.min(g.pages_y - 1),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vt_stream::projection_scale;

    /// The shadow pages' convention **is** the camera's, by identity rather than
    /// by two constants that agree today.
    #[test]
    fn a_shadow_page_uses_the_cameras_depth_convention() {
        assert_eq!(VSM_DEPTH_CLEAR, DEPTH_CLEAR);
        assert_eq!(VSM_DEPTH_COMPARE, DEPTH_COMPARE);
        assert_eq!(VSM_DEPTH_CLEAR, 0.0);
        assert_eq!(VSM_DEPTH_COMPARE, wgpu::CompareFunction::Greater);
        // …and the CSM's forward-Z exception is UNTOUCHED: P27.5 demotes that
        // path, this batch does not edit it.
        assert_eq!(
            crate::passes::shadow::SHADOW_DEPTH_CLEAR,
            1.0,
            "the CSM's forward-Z clear moved — the demotion is P27.5's, not this \
             batch's"
        );
    }

    /// The smallest world-space step at `z` metres along the light that changes
    /// the stored f32 depth — **the measurement**, taken through the shipped
    /// matrix rather than through a formula beside it.
    ///
    /// Bisected rather than differentiated: what a shadow map can resolve is
    /// literally "how far must a surface move before the depth buffer notices",
    /// and a derivative would be a statement about the algebra.
    fn depth_step_m(vp: &Mat4, fwd: Vec3, origin: Vec3, z: f64) -> f64 {
        let d = |t: f64| {
            let p = origin + fwd * (z + t) as f32;
            let c = *vp * p.extend(1.0);
            c.z / c.w
        };
        let base = d(0.0);
        let (mut lo, mut hi) = (0.0f64, 1.0f64);
        // Grow until the depth moves at all (a range where it never does is a
        // range this projection cannot represent, and the caller asserts on it).
        while d(hi) == base && hi < 1e6 {
            hi *= 2.0;
        }
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            if d(mid) == base {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        hi
    }

    /// Forward-Z ortho over `[near, far]`, RH view space — the CSM's convention,
    /// built here so the two can be measured against each other.
    fn ortho_forward_z(half: f32, near: f32, far: f32) -> Mat4 {
        let inv = 1.0 / (far - near);
        Mat4::from_cols(
            glam::Vec4::new(1.0 / half, 0.0, 0.0, 0.0),
            glam::Vec4::new(0.0, 1.0 / half, 0.0, 0.0),
            glam::Vec4::new(0.0, 0.0, -inv, 0.0),
            glam::Vec4::new(0.0, 0.0, -near * inv, 1.0),
        )
    }

    /// Forward-Z perspective over `[near, far]`, RH view space (the DirectX
    /// depth range) — the convention a forward-Z spot shadow would have used.
    fn perspective_forward_z(fov_y: f32, near: f32, far: f32) -> Mat4 {
        let f = 1.0 / (fov_y * 0.5).tan();
        let inv = 1.0 / (far - near);
        Mat4::from_cols(
            glam::Vec4::new(f, 0.0, 0.0, 0.0),
            glam::Vec4::new(0.0, f, 0.0, 0.0),
            glam::Vec4::new(0.0, 0.0, -far * inv, -1.0),
            glam::Vec4::new(0.0, 0.0, -far * near * inv, 0.0),
        )
    }

    /// **THE DEPTH-CONVENTION MEASUREMENT** (P27.1 clause 4), and the numbers
    /// `docs/memos/p27-1-depth-convention.md` quotes.
    ///
    /// Two frusta, two conventions each, worst-case depth resolution over the
    /// whole range. The ruling is not "reverse-Z is nicer": it is that the
    /// orthographic case cannot tell the two apart and the perspective case —
    /// which P27 introduces and the CSM never had — prefers reverse-Z by three
    /// and a half orders of magnitude.
    #[test]
    fn the_depth_convention_is_decided_by_the_perspective_case() {
        let origin = Vec3::ZERO;
        let fwd = Vec3::NEG_Z;
        let worst = |vp: &Mat4, lo: f64, hi: f64| {
            let mut w = 0.0f64;
            let mut at = lo;
            for i in 0..=400 {
                let z = lo + (hi - lo) * (i as f64 / 400.0);
                let s = depth_step_m(vp, fwd, origin, z);
                if s > w {
                    w = s;
                    at = z;
                }
            }
            (w, at)
        };

        // ── orthographic, 200 m of range: a TIE, to three significant figures ──
        let (near, far) = (1e-3f32, 200.0f32);
        let rev = ortho_reverse_z(32.0, 1.0, near, far);
        let fwd_z = ortho_forward_z(32.0, near, far);
        let (rev_worst, rev_at) = worst(&rev, 0.01, 199.9);
        let (fwd_worst, fwd_at) = worst(&fwd_z, 0.01, 199.9);
        // Ranges rather than equalities, and not from timidity: the projections
        // are built with `f32::tan`, which the P14 law records is not
        // bit-portable across platforms, so an exact pin would be a bound that
        // reddens CI on one leg. What the ruling rests on is the RATIO, and the
        // ratio is nowhere near the tolerance.
        assert!(
            (1.4e-5..2.6e-5).contains(&rev_worst) && (1.0e-5..2.0e-5).contains(&fwd_worst),
            "the ortho measurement moved: reverse {rev_worst:e} m, forward \
             {fwd_worst:e} m — the memo quotes 21.2 µm and 15.2 µm"
        );
        assert!(
            rev_worst / fwd_worst < 2.0,
            "the two ortho conventions are {:.2}× apart — the ruling says the \
             orthographic case cannot tell them apart",
            rev_worst / fwd_worst
        );
        // …and NEITHER peak is at an end of the box. The intuition that forward-Z
        // is coarse at the far plane and reverse-Z at the near one is wrong: f32's
        // ULP is a step function of the STORED depth, so a linear depth's coarsest
        // step lands wherever the [0.5, 1) binade does. This is the assertion that
        // caught the first draft of this module's own doc comment.
        assert!(
            (10.0..190.0).contains(&rev_at) && (10.0..190.0).contains(&fwd_at),
            "a peak moved to an end of the box: reverse at {rev_at} m, forward at \
             {fwd_at} m"
        );

        // ── perspective, a 0.1 m → 50 m spot: NOT a tie ──
        let fov = 60f32.to_radians();
        let rev_p = glam::camera::rh::proj::directx::perspective_infinite_reverse(fov, 1.0, 0.1);
        let fwd_p = perspective_forward_z(fov, 0.1, 50.0);
        let (rev_pw, _) = worst(&rev_p, 0.1, 49.9);
        let (fwd_pw, fwd_pat) = worst(&fwd_p, 0.1, 49.9);
        assert!(
            (5.0e-6..1.0e-5).contains(&rev_pw),
            "reverse-infinite worst case moved: {rev_pw:e} m (the memo quotes 7.43 µm)"
        );
        assert!(
            (1.4e-3..2.4e-3).contains(&fwd_pw),
            "forward worst case moved: {fwd_pw:e} m (the memo quotes 1.87 mm)"
        );
        let ratio = fwd_pw / rev_pw;
        assert!(
            ratio > 150.0,
            "forward-Z is only {ratio:.0}× coarser — the ruling rests on ~251×"
        );
        // Forward-Z spends it at the FAR end, which is where a spot's shadow is
        // biggest on screen — and unlike the ortho case above, this end IS where
        // the intuition puts it, because a perspective depth is not linear.
        assert!(fwd_pat > 40.0, "forward-Z's worst case is at {fwd_pat} m");
    }

    /// The level rule: one shadow texel per screen pixel, erring **coarse**,
    /// clamped into the tree at both ends. The function `vsm_mark.wgsl` mirrors,
    /// pinned as a table rather than described.
    #[test]
    fn the_justified_level_is_one_texel_per_pixel_rounded_coarse() {
        // A 1 cm shadow texel under a 1 cm pixel: level 0.
        assert_eq!(vsm_justified_level(0.01, 0.01, 8), 0);
        // Twice as coarse a pixel: one level up.
        assert_eq!(vsm_justified_level(0.01, 0.02, 8), 1);
        assert_eq!(vsm_justified_level(0.01, 0.08, 8), 3);
        // A distant pixel: clamped at the coarsest rather than running off.
        assert_eq!(vsm_justified_level(0.01, 1000.0, 8), 7);
        // Finer than the finest level: level 0, never negative.
        assert_eq!(vsm_justified_level(0.01, 0.0001, 8), 0);
        // Erring coarse: 1.5 texels of pixel is level 1, not level 0.
        assert_eq!(vsm_justified_level(0.01, 0.015, 8), 1);
        // A one-level tree has one answer, and a zero-level one cannot divide.
        assert_eq!(vsm_justified_level(0.01, 100.0, 1), 0);
        assert_eq!(vsm_justified_level(0.01, 100.0, 0), 0);
        assert_eq!(vsm_justified_level(0.0, 1.0, 8), 7, "no zero division");
    }

    /// A clipmap point is served by the finest level that **contains** it, never
    /// by a finer one its footprint would justify — and by nothing at all when it
    /// is outside the whole clipmap.
    #[test]
    fn a_clipmap_point_cannot_be_served_by_a_level_that_misses_it() {
        assert_eq!(clipmap_containing_level(0.5, 8), Some(0));
        assert_eq!(clipmap_containing_level(1.0, 8), Some(0));
        assert_eq!(clipmap_containing_level(1.5, 8), Some(1));
        assert_eq!(clipmap_containing_level(2.0, 8), Some(1));
        assert_eq!(clipmap_containing_level(2.1, 8), Some(2));
        assert_eq!(clipmap_containing_level(128.0, 8), Some(7));
        assert_eq!(
            clipmap_containing_level(300.0, 8),
            None,
            "past the coarsest level there is no shadow data, and that is the \
             answer rather than a clamp onto the edge"
        );
        assert_eq!(clipmap_containing_level(0.1, 0), None);
    }

    fn view_at(z: f64) -> RenderView {
        RenderView {
            origin: inf_math::FloatingOrigin::new(glam::DVec3::ZERO),
            eye_world: glam::DVec3::new(0.0, 0.0, z),
            forward: Vec3::NEG_Z,
            up: Vec3::Y,
            fov_y: 60f32.to_radians(),
            near: 0.05,
            width: 1920,
            height: 1080,
            ortho: None,
        }
    }

    /// The clipmap centre is **snapped**, so sub-page camera motion does not move
    /// a single page — the property P27.3's caching clause rests on.
    #[test]
    fn the_clipmap_centre_does_not_move_under_sub_page_motion() {
        let page = clipmap_page_world(32.0, 64);
        assert!(
            (page - 1.0).abs() < 1e-6,
            "64 pages over 64 m is 1 m a page"
        );
        let a = clipmap_centre(Vec3::new(10.2, 0.0, -3.4), page);
        let b = clipmap_centre(Vec3::new(10.2 + 0.2 * page, 0.0, -3.4), page);
        assert_eq!(a, b, "a fifth of a page moved the clipmap");
        assert_eq!(a, Vec3::new(10.0, 0.0, -3.0));
        // …and a whole page's motion DOES move it, or the snap is a constant.
        let c = clipmap_centre(Vec3::new(10.2 + page, 0.0, -3.4), page);
        assert_ne!(a, c);
        assert_eq!(c - a, Vec3::new(page, 0.0, 0.0));
        // The property behind those two cases, swept rather than sampled: over a
        // 20-page walk the centre only ever takes page-multiple values, and it
        // takes about one per page rather than one per step. (0.2 of a page is
        // not arbitrary: 0.3 from 10.2 crosses the 10.5 rounding boundary, which
        // is what the first draft of this arm measured and asserted was a bug.)
        let mut seen = std::collections::BTreeSet::new();
        for i in 0..200 {
            let c = clipmap_centre(Vec3::new(i as f32 * 0.1 * page, 0.0, 0.0), page);
            assert_eq!((c.x / page).fract(), 0.0, "{c:?} is off the lattice");
            seen.insert(c.x.to_bits());
        }
        assert!(
            (11..=21).contains(&seen.len()),
            "a 20-page walk in tenths produced {} distinct centres",
            seen.len()
        );
    }

    /// **The CPU twin marks the page the geometry says it should**, on a
    /// clipmap — a hand-computed case, so the twin is checked against arithmetic
    /// rather than against itself.
    #[test]
    fn the_clipmap_twin_marks_the_page_the_geometry_names() {
        let settings = VsmSettings {
            clipmap_pages_per_side: 8,
            clipmap_levels: 4,
            first_level_extent_m: 32.0,
            ..Default::default()
        };
        // The tree the settings name, so the fixture and the shipped builder
        // cannot describe two different clipmaps.
        let desc = VsmLightDesc::clipmap(settings.clipmap_levels, settings.clipmap_pages_per_side);
        let half = settings.first_level_extent_m;
        // A sun along +Z, clipmap centred on the origin — chosen because that is
        // the one direction whose light basis is unambiguous: `clipmap_matrix`
        // picks `up = +Y`, so its right is `+X` and its up is `+Y`, and the page
        // a world point lands on is hand-computable. (A sun straight overhead is
        // NOT: the basis degenerates onto `up = +Z` and the right axis comes out
        // `-X`, so "world +x is page +x" is false — measured, when the first
        // draft of this arm asserted it.)
        let vp = clipmap_matrix(Vec3::Z, Vec3::ZERO, half, 4.0 * half);
        let page0 = clipmap_page_world(half, settings.clipmap_pages_per_side);
        assert_eq!(page0, 8.0, "8 pages over 64 m");
        let texels0 = (desc.levels[0].pages_x * VSM_PAGE_SIZE) as f32;
        let proj = VsmProjection {
            view_proj: vp.to_cols_array(),
            info: [0, 0, 0, VSM_PROJ_ORTHO],
            light: [0.0, 0.0, 0.0, 2.0 * half / texels0],
        };
        // A pixel whose footprint is finer than a level-0 texel: level 0.
        let texel0 = 2.0 * half / texels0;
        let p = mark_page_for(&proj, &desc, Vec3::new(1.0, 1.0, 0.0), texel0 * 0.5)
            .expect("inside the clipmap");
        assert_eq!(p.level, 0);
        // Hand-computed: the box is ±32 m over 8 pages, so a point 1 m right of
        // and 1 m above the centre is at NDC (1/32, 1/32) = (0.03125, 0.03125).
        // The column is `(0.5 + 0.015625) × 8 = 4.125` → **4**; the row is
        // `(0.5 − 0.015625) × 8 = 3.875` → **3**, because NDC y runs UP and the
        // page grid's y runs DOWN. That asymmetry is the flip, and it is the one
        // thing in this file the shader has to copy exactly.
        assert_eq!((p.x, p.y), (4, 3), "{p:?}");
        // …and the centre itself lands on the page boundary, both axes at 4.
        let c = mark_page_for(&proj, &desc, Vec3::ZERO, texel0 * 0.5).expect("the centre");
        assert_eq!((c.x, c.y), (4, 4), "{c:?}");
        // …and one metre the other way is one page the other way, on each axis
        // independently — which is what makes the flip a claim about y alone.
        let n =
            mark_page_for(&proj, &desc, Vec3::new(-1.0, -1.0, 0.0), texel0 * 0.5).expect("inside");
        assert_eq!((n.x, n.y), (3, 4), "{n:?}");
        // A point outside level 0 but inside level 1 is served by level 1.
        let q = mark_page_for(&proj, &desc, Vec3::new(50.0, 0.0, 0.0), texel0 * 0.5)
            .expect("inside level 1");
        assert_eq!(q.level, 1, "{q:?}");
        // …and a point outside the whole clipmap marks nothing.
        assert_eq!(
            mark_page_for(&proj, &desc, Vec3::new(5000.0, 0.0, 0.0), texel0 * 0.5),
            None
        );
        // A coarse pixel takes a coarse level even at the centre.
        let r = mark_page_for(&proj, &desc, Vec3::ZERO, texel0 * 100.0).expect("centre");
        assert_eq!(r.level, 3, "a 100-texel pixel justifies the coarsest level");
    }

    /// Every shadow-casting light gets the tree its kind names, a point light
    /// costs six projections, and the projection cap drops a light whole.
    #[test]
    fn each_light_kind_gets_its_own_tree_and_the_cap_drops_a_light_whole() {
        use crate::scene::RenderLight;
        let mut scene = RenderScene::default();
        let settings = VsmSettings::default();
        scene.lights.push(RenderLight {
            kind: LightKind::Directional,
            direction: Vec3::Y,
            ..Default::default()
        });
        scene.lights.push(RenderLight {
            kind: LightKind::Spot,
            cast_shadows: true,
            ..Default::default()
        });
        scene.lights.push(RenderLight {
            kind: LightKind::Point,
            cast_shadows: true,
            ..Default::default()
        });
        // …and one that does not cast, which must contribute nothing at all.
        scene.lights.push(RenderLight {
            kind: LightKind::Point,
            cast_shadows: false,
            ..Default::default()
        });
        let trees = vsm_light_trees(&scene, &settings);
        assert_eq!(trees.len(), 3, "a non-casting light took a tree");
        assert_eq!(trees[0].kind, VsmTreeKind::Clipmap);
        assert_eq!(trees[1].kind, VsmTreeKind::Quadtree);
        assert_eq!(trees[2].kind, VsmTreeKind::Cube);

        let blocks = vec![0u32; trees.len()];
        let bases = vec![0u32; trees.len()];
        let projections =
            vsm_projections(&scene, &view_at(10.0), &settings, &trees, &blocks, &bases);
        assert_eq!(projections.len(), 1 + 1 + 6, "a cube is six projections");
        assert_eq!(projections[0].info[3], VSM_PROJ_ORTHO);
        assert_eq!(projections[1].info[3], VSM_PROJ_PERSPECTIVE);
        for (i, p) in projections[2..].iter().enumerate() {
            assert_eq!(p.info[2], i as u32, "the faces are in order");
        }

        // The cap counts PROJECTIONS: eleven point lights is 66 > 64, so the
        // eleventh is dropped whole rather than given four faces.
        let mut many = RenderScene::default();
        for _ in 0..11 {
            many.lights.push(RenderLight {
                kind: LightKind::Point,
                cast_shadows: true,
                ..Default::default()
            });
        }
        let trees = vsm_light_trees(&many, &settings);
        assert_eq!(trees.len(), 10, "10 × 6 = 60 fits, 11 × 6 = 66 does not");

        // **THE PREFIX INVARIANT.** A sun after the eleventh point light would
        // FIT — one projection into the four that are left — and it must still be
        // dropped, because handle `n` is the `n`-th shadow-casting light in scene
        // order and `vsm_projections` rebuilds that by re-walking `scene.lights`.
        // A cap that skipped instead of stopping would make that sun handle 10,
        // give it the point light's table block and bit range, and mark its
        // clipmap into another light's address space — no error, no counter, and
        // an atlas full of pages nothing reads.
        many.lights.push(RenderLight {
            kind: LightKind::Directional,
            cast_shadows: true,
            ..Default::default()
        });
        let trees = vsm_light_trees(&many, &settings);
        assert_eq!(
            trees.len(),
            10,
            "a light that fits was admitted PAST one that did not — the handle \
             mapping is no longer the scene order"
        );
        assert!(
            trees.iter().all(|t| t.kind == VsmTreeKind::Cube),
            "the sun took a handle behind the light that stopped the list"
        );
        // …and the projections agree: ten cubes, sixty entries, no clipmap.
        let blocks = vec![0u32; trees.len()];
        let bases = vec![0u32; trees.len()];
        let ps = vsm_projections(&many, &view_at(10.0), &settings, &trees, &blocks, &bases);
        assert_eq!(ps.len(), 60);
        assert!(ps.iter().all(|p| p.info[3] == VSM_PROJ_PERSPECTIVE));
    }

    /// A **spot** projection points down its beam and its level-0 texel scales
    /// with distance from the light, which is what makes one rule serve both
    /// families.
    #[test]
    fn a_spot_projection_scales_its_texel_with_distance() {
        let settings = VsmSettings::default();
        let desc = VsmLightDesc::quadtree(settings.spot_levels);
        let texels0 = (desc.levels[0].pages_x * VSM_PAGE_SIZE) as f32;
        let outer = 40f32.to_radians().cos();
        let per_metre = 2.0 * (spot_fov_y(outer) * 0.5).tan() / texels0;
        let proj = VsmProjection {
            // The light at the origin shining down -Z.
            view_proj: spot_matrix(Vec3::ZERO, Vec3::Z, outer, 0.1).to_cols_array(),
            info: [0, 0, 0, VSM_PROJ_PERSPECTIVE],
            light: [0.0, 0.0, 0.0, per_metre],
        };
        // The same pixel footprint at 4 m and at 32 m. A spot's page grid is
        // ANGULAR, so a level-0 texel near the light is tiny and one far away is
        // large — which means a near receiver is served by a COARSER level (level
        // 0 is far more resolution than its pixel needs) and a far one by a finer
        // level. That is the opposite of the directional intuition and it is the
        // direction the arithmetic actually has; the first draft of this arm
        // asserted the intuition and measured 5 against 2.
        let near =
            mark_page_for(&proj, &desc, Vec3::new(0.0, 0.0, -4.0), 0.02).expect("in the cone");
        let far =
            mark_page_for(&proj, &desc, Vec3::new(0.0, 0.0, -32.0), 0.02).expect("in the cone");
        assert!(
            near.level > far.level,
            "a receiver 8× closer to the light took level {} against {} — the \
             angular texel did not scale with distance",
            near.level,
            far.level
        );
        // ANTI-VACUITY: neither answer is at an end of the tree, so this is a
        // statement about the rule and not about two clamps.
        assert!(
            far.level > 0 && near.level + 1 < desc.level_count(),
            "the fixture's levels ({}, {}) sit at the ends of a {}-level tree",
            near.level,
            far.level,
            desc.level_count()
        );
        // Behind the light: nothing.
        assert_eq!(
            mark_page_for(&proj, &desc, Vec3::new(0.0, 0.0, 5.0), 0.02),
            None
        );
        // Outside the cone: nothing.
        assert_eq!(
            mark_page_for(&proj, &desc, Vec3::new(100.0, 0.0, -2.0), 0.02),
            None
        );
        // The cone's degenerate ends do not produce a degenerate matrix.
        assert!(spot_fov_y(1.0) >= 2.0_f32.to_radians());
        assert!(spot_fov_y(-1.0) <= 170.0_f32.to_radians());
        assert!(spot_matrix(Vec3::ZERO, Vec3::Z, 1.0, 0.1)
            .to_cols_array()
            .iter()
            .all(|f| f.is_finite()));
    }

    /// **A perspective light's level-0 texel is the width its own frustum
    /// covers** — the two `per_metre` constants [`vsm_projections`] writes into
    /// `VsmProjection::light.w`, checked against the shipped matrices instead of
    /// restated.
    ///
    /// The P27.1 audit found both unpinned: halving the cube face's `2.0` moves
    /// every marked level by exactly one and the point-light arm's
    /// anti-vacuity band absorbs it, so the number that decides how much shadow
    /// resolution a point light gets had no arm at all.
    #[test]
    fn a_perspective_lights_texel_is_the_width_its_frustum_covers() {
        let settings = VsmSettings::default();
        let d = 8.0f32;

        // ── a cube face: 90°, so the cross-section at `d` is `2 · d · tan 45°`.
        let cube = VsmLightDesc::cube(settings.point_levels);
        let cube_texels = (cube.levels[0].pages_x * VSM_PAGE_SIZE) as f32;
        let cube_per_metre = 2.0 / cube_texels;
        // Face 4 is `+Z`; a point `w` metres off its axis at `d` metres out.
        let m = cube_face_matrix(Vec3::ZERO, 4, settings.perspective_near_m);
        let ndc_x = |w: f32| {
            let c = m * Vec3::new(w, 0.0, d).extend(1.0);
            c.x / c.w
        };
        // One level-0 texel is `2 / texels` of NDC, since NDC spans [-1, 1].
        assert!(
            ((ndc_x(cube_per_metre * d) - ndc_x(0.0)).abs() - 2.0 / cube_texels).abs() < 1e-5,
            "a cube face's per-metre texel does not span one texel of its own \
             projection at {d} m"
        );

        // ── a spot: the same claim at the outer cone's own field of view.
        let outer = 40f32.to_radians().cos();
        let spot = VsmLightDesc::quadtree(settings.spot_levels);
        let spot_texels = (spot.levels[0].pages_x * VSM_PAGE_SIZE) as f32;
        let spot_per_metre = 2.0 * (spot_fov_y(outer) * 0.5).tan() / spot_texels;
        let s = spot_matrix(Vec3::ZERO, Vec3::Z, outer, settings.perspective_near_m);
        let s_ndc_x = |w: f32| {
            let c = s * Vec3::new(w, 0.0, -d).extend(1.0);
            c.x / c.w
        };
        assert!(
            ((s_ndc_x(spot_per_metre * d) - s_ndc_x(0.0)).abs() - 2.0 / spot_texels).abs() < 1e-5,
            "a spot's per-metre texel does not span one texel of its own cone"
        );

        // …and the SHIPPED builder writes exactly those numbers, so the two
        // cannot drift apart.
        use crate::scene::RenderLight;
        let mut scene = RenderScene::default();
        scene.lights.push(RenderLight {
            kind: LightKind::Spot,
            outer_cos: outer,
            cast_shadows: true,
            ..Default::default()
        });
        scene.lights.push(RenderLight {
            kind: LightKind::Point,
            cast_shadows: true,
            ..Default::default()
        });
        let trees = vsm_light_trees(&scene, &settings);
        let ps = vsm_projections(
            &scene,
            &view_at(10.0),
            &settings,
            &trees,
            &vec![0u32; trees.len()],
            &vec![0u32; trees.len()],
        );
        assert_eq!(ps.len(), 7, "a spot and a cube");
        assert!((ps[0].light[3] - spot_per_metre).abs() < 1e-9, "the spot");
        for p in &ps[1..] {
            assert!((p.light[3] - cube_per_metre).abs() < 1e-9, "a cube face");
        }
        // ANTI-VACUITY: the two lights' texels really are different numbers, so
        // the two assertions above are not one assertion twice.
        assert_ne!(spot_per_metre, cube_per_metre);
    }

    /// **The jittered-inverse fix, measured — and the precise bound on what any
    /// arm may be asked to prove about it.**
    ///
    /// `VsmMarker::record` is handed `jvp.inverse()`: the inverse of the matrix
    /// the frame's depth was actually rasterized with, jitter and all. Feeding it
    /// `view.view_proj().inverse()` instead is not a small error in one place, it
    /// is a *systematic* one — every pixel in the frame reconstructs off its own
    /// surface — and the first half of this arm says by exactly how much.
    ///
    /// The second half says why the marked **set** cannot see it, which is the
    /// ledger's "this fix has no falsifying arm", made a measurement rather than
    /// a claim. The displacement is `|jitter| × pixel_world`, at most half a
    /// pixel. The level rule targets one shadow texel per screen pixel and errs
    /// **coarse**, so a page at the level a pixel justifies is at least 128
    /// pixels across: the displacement is at most **1/256 of a page**. A page can
    /// therefore only enter or leave the marked set if the geometry's coverage of
    /// it is already thinner than half a pixel — and in that configuration the
    /// *correct* reconstruction is equally jitter-dependent, because the
    /// rasterized coverage moves with the jitter too. The bound is structural, not
    /// a property of this fixture.
    #[test]
    fn the_unjittered_inverse_displaces_a_pixel_without_moving_its_page() {
        let v = view_at(6.0);
        let base = v.view_proj();
        // The renderer's own jitter, at the frame of the 16-frame cycle whose
        // Halton offset is largest — the worst case, not an average one.
        let jitter = (0..16u64)
            .map(crate::settings::halton_jitter)
            .max_by(|a, b| (a[0] * a[0] + a[1] * a[1]).total_cmp(&(b[0] * b[0] + b[1] * b[1])))
            .expect("a sixteen-frame cycle");
        let jvp = Mat4::from_translation(Vec3::new(
            2.0 * jitter[0] / v.width as f32,
            2.0 * jitter[1] / v.height as f32,
            0.0,
        )) * base;

        // A surface point the rasterizer wrote depth for, and the clip position
        // it wrote it at.
        let surface = Vec3::new(0.3, 1.1, 0.0);
        let clip = jvp * surface.extend(1.0);
        let ndc = clip.truncate() / clip.w;
        // What the shader does with the inverse it is handed.
        let recover = |inv: Mat4| {
            let h = inv * ndc.extend(1.0);
            h.truncate() / h.w
        };
        let right = recover(jvp.inverse());
        let wrong = recover(base.inverse());
        assert!(
            (right - surface).length() < 1e-3,
            "the matrix the depth was drawn with did not recover the point that \
             drew it: {right:?}"
        );

        let pixel_world = (surface - v.eye_local()).length() / projection_scale(&v);
        let displaced = (wrong - surface).length();
        let in_pixels = displaced / pixel_world;
        let jitter_px = (jitter[0] * jitter[0] + jitter[1] * jitter[1]).sqrt();
        assert!(
            (in_pixels - jitter_px).abs() < 0.02 * jitter_px.max(1e-3),
            "the displacement is {in_pixels} pixels where the jitter is {jitter_px}"
        );
        assert!(
            in_pixels > 0.1 && in_pixels < 0.71,
            "the defect displaced {in_pixels} pixels — it is neither absent nor \
             more than half a pixel on the diagonal"
        );

        // THE BOUND: the same two reconstructions mark the same page, so no arm
        // over a marked set can tell them apart.
        let settings = VsmSettings {
            clipmap_pages_per_side: 8,
            clipmap_levels: 6,
            first_level_extent_m: 6.0,
            ..Default::default()
        };
        let desc = VsmLightDesc::clipmap(settings.clipmap_levels, settings.clipmap_pages_per_side);
        let texels0 = (desc.levels[0].pages_x * VSM_PAGE_SIZE) as f32;
        let proj = VsmProjection {
            view_proj: clipmap_matrix(
                Vec3::Z,
                clipmap_centre(
                    v.eye_local(),
                    clipmap_page_world(
                        settings.first_level_extent_m,
                        settings.clipmap_pages_per_side,
                    ),
                ),
                settings.first_level_extent_m,
                8.0 * settings.first_level_extent_m,
            )
            .to_cols_array(),
            info: [0, 0, 0, VSM_PROJ_ORTHO],
            light: [0.0, 0.0, 0.0, 2.0 * settings.first_level_extent_m / texels0],
        };
        let page_of = |p: Vec3| mark_page_for(&proj, &desc, p, pixel_world);
        let marked = page_of(right).expect("the fixture's point marks a page");
        assert_eq!(
            page_of(wrong),
            Some(marked),
            "the displaced reconstruction marked a different page — if this ever \
             fails, the fix HAS a falsifying arm and the ledger's remainder is stale"
        );
        // …and the reason, as arithmetic: the page is two orders of magnitude
        // wider than the error.
        let page_world = 2.0 * settings.first_level_extent_m * (1u32 << marked.level) as f32
            / desc.levels[0].pages_x as f32;
        assert!(
            displaced * 128.0 < page_world,
            "a {displaced} m displacement against a {page_world} m page is not the \
             1/256-of-a-page bound the ledger rests on"
        );
    }

    /// **A page's projection contains exactly the world points that mark it.**
    ///
    /// The P27.2 clause that everything else rests on: the raster fills a page with
    /// the casters `vsm_page_matrix` admits, and the marking pass allocated that
    /// page because [`mark_page_for`] said a pixel needed it. If those two
    /// statements are not the same statement, the atlas holds depth from the wrong
    /// ground and every receiver is wrong in a way no counter reports.
    ///
    /// Swept rather than sampled, and asserted in **both** directions — a
    /// containment test that only ever says "yes" is satisfied by a matrix that
    /// maps the whole world into the unit square.
    #[test]
    fn a_pages_projection_contains_exactly_the_points_that_mark_it() {
        let settings = VsmSettings {
            clipmap_pages_per_side: 8,
            clipmap_levels: 4,
            first_level_extent_m: 32.0,
            ..Default::default()
        };
        let desc = VsmLightDesc::clipmap(settings.clipmap_levels, settings.clipmap_pages_per_side);
        let half = settings.first_level_extent_m;
        let vp = clipmap_matrix(Vec3::Z, Vec3::ZERO, half, 4.0 * half);
        let texels0 = (desc.levels[0].pages_x * VSM_PAGE_SIZE) as f32;
        let proj = VsmProjection {
            view_proj: vp.to_cols_array(),
            info: [0, 0, 0, VSM_PROJ_ORTHO],
            light: [0.0, 0.0, 0.0, 2.0 * half / texels0],
        };
        let texel0 = 2.0 * half / texels0;
        // Fine enough a footprint that the level rule always answers 0, so the
        // sweep is a statement about the PAGE arithmetic and not about the level
        // rule (which its own arm covers).
        let pixel = texel0 * 0.5;
        let (mut inside, mut outside) = (0u32, 0u32);
        let mut pages_hit = std::collections::BTreeSet::new();
        for iy in 0..40 {
            for ix in 0..40 {
                let p = Vec3::new(
                    -half + 2.0 * half * (ix as f32 + 0.5) / 40.0,
                    -half + 2.0 * half * (iy as f32 + 0.5) / 40.0,
                    0.0,
                );
                let marked = mark_page_for(&proj, &desc, p, pixel).expect("inside level 0");
                assert_eq!(marked.level, 0, "the sweep left level 0 at {p:?}");
                pages_hit.insert((marked.x, marked.y));
                // Its own page contains it…
                let g = desc.levels[marked.level as usize];
                let m = vsm_page_matrix(
                    vp,
                    desc.kind,
                    marked.level,
                    g.pages_x,
                    g.pages_y,
                    marked.x,
                    marked.y,
                );
                let c = m * p.extend(1.0);
                let n = c.truncate() / c.w;
                assert!(
                    n.x.abs() <= 1.0 + 1e-4 && n.y.abs() <= 1.0 + 1e-4,
                    "{p:?} marks {marked:?} but its own page's projection puts it at \
                     {n:?}"
                );
                inside += 1;
                // …and its NEIGHBOURS do not. This is the half that fails when the
                // page centre, the level scale or the vertical flip is wrong, and
                // it is the half a one-directional test cannot see.
                for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                    let (nx, ny) = (marked.x as i32 + dx, marked.y as i32 + dy);
                    if nx < 0 || ny < 0 || nx >= g.pages_x as i32 || ny >= g.pages_y as i32 {
                        continue;
                    }
                    let m = vsm_page_matrix(
                        vp,
                        desc.kind,
                        marked.level,
                        g.pages_x,
                        g.pages_y,
                        nx as u32,
                        ny as u32,
                    );
                    let c = m * p.extend(1.0);
                    let n = c.truncate() / c.w;
                    assert!(
                        n.x.abs() > 1.0 - 1e-4 || n.y.abs() > 1.0 - 1e-4,
                        "{p:?} marks ({}, {}) and page ({nx}, {ny}) also contains it \
                         at {n:?}",
                        marked.x,
                        marked.y
                    );
                    outside += 1;
                }
            }
        }
        // ANTI-VACUITY on all three counters: the sweep really did cover many
        // pages, really did test both directions, and really did reject.
        assert_eq!(inside, 1_600);
        assert!(outside > 5_000, "only {outside} neighbour rejections");
        assert_eq!(
            pages_hit.len(),
            64,
            "the sweep covered {} of 64 pages",
            pages_hit.len()
        );
    }

    /// A **coarse** clipmap level's pages cover four times the ground of the level
    /// below, concentrically — the `1/2^L` half of the rule, which the level-0
    /// sweep above cannot see at all.
    #[test]
    fn a_coarser_clipmap_page_covers_twice_the_extent_on_each_axis() {
        let half = 32.0f32;
        let vp = clipmap_matrix(Vec3::Z, Vec3::ZERO, half, 8.0 * half);
        let n = 8u32;
        // The centre page of an even grid: `N/2` is the first page right of centre,
        // so page (4, 3) is the one straddling the origin upward — the same page
        // `the_clipmap_twin_marks_the_page_the_geometry_names` hand-computes.
        let world_span = |level: u32| {
            let m = vsm_page_matrix(vp, VsmTreeKind::Clipmap, level, n, n, 4, 3);
            // Walk +x until the page's own NDC leaves the unit square: that
            // distance is the page's half-width in metres.
            let mut lo = 0.0f32;
            let mut hi = 4.0 * half;
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                let c = m * Vec3::new(mid, 1.0, 0.0).extend(1.0);
                if (c.x / c.w).abs() <= 1.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            lo
        };
        let l0 = world_span(0);
        let l1 = world_span(1);
        let l2 = world_span(2);
        // 64 m over 8 pages is an 8 m page — and the walk starts at `x = 0`, which
        // for an EVEN grid is the clipmap centre and therefore page 4's own left
        // edge, so what it measures is the page's whole width rather than half of
        // it. (Measured: the first draft of this arm asserted 4 m and read 8.)
        assert!((l0 - 8.0).abs() < 0.05, "level 0 page width {l0} m");
        assert!((l1 / l0 - 2.0).abs() < 0.01, "{l1} / {l0}");
        assert!((l2 / l1 - 2.0).abs() < 0.01, "{l2} / {l1}");
        // A **quadtree** does not do that: every level covers the light's whole
        // frustum, so the page grid alone decides the page's size.
        let spot = spot_matrix(Vec3::ZERO, Vec3::Z, 40f32.to_radians().cos(), 0.05);
        let q_span = |level: u32, pages: u32| {
            let m = vsm_page_matrix(spot, VsmTreeKind::Quadtree, level, pages, pages, 0, 0);
            let c = m * Vec3::new(0.0, 0.0, -10.0).extend(1.0);
            (c.x / c.w, c.y / c.w)
        };
        // The beam axis at 10 m is the level's centre, so under a 2×2 grid it sits
        // at the inner corner of page (0, 0) — at NDC (+1, −1) — whatever the level.
        for (level, pages) in [(0u32, 2u32), (1, 2)] {
            let (x, y) = q_span(level, pages);
            assert!(
                (x - 1.0).abs() < 1e-3 && (y + 1.0).abs() < 1e-3,
                "level {level}: the beam axis landed at ({x}, {y}), not the page corner"
            );
        }
    }

    /// The page cull keeps what a page can see and drops what it cannot — **and
    /// survives a projection with no far plane**, which every spot and every cube
    /// face is.
    #[test]
    fn the_page_cull_rejects_a_sphere_the_page_cannot_see() {
        let half = 32.0f32;
        let vp = clipmap_matrix(Vec3::Z, Vec3::ZERO, half, 4.0 * half);
        let n = 8u32;
        let page = |x, y| vsm_page_matrix(vp, VsmTreeKind::Clipmap, 0, n, n, x, y);
        // An 8 m page: a 0.1 m sphere at its centre is seen, the same sphere three
        // pages away is not, and a 20 m sphere three pages away IS (it reaches).
        let centre_of = |x: u32, y: u32| {
            Vec3::new(
                -half + 8.0 * (x as f32 + 0.5),
                half - 8.0 * (y as f32 + 0.5),
                0.0,
            )
        };
        assert!(vsm_page_sees_sphere(&page(4, 3), centre_of(4, 3), 0.1));
        assert!(!vsm_page_sees_sphere(&page(4, 3), centre_of(7, 3), 0.1));
        assert!(vsm_page_sees_sphere(&page(4, 3), centre_of(7, 3), 20.0));
        // Behind the light's box, along the light: rejected by the depth planes,
        // which is the half a 2-D test would miss.
        assert!(!vsm_page_sees_sphere(
            &page(4, 3),
            centre_of(4, 3) - Vec3::Z * 500.0,
            1.0
        ));

        // …and the infinite-far case. A cube face's `w − z` row is identically
        // zero; trusting it would reject every caster of every point light.
        let face = cube_face_matrix(Vec3::ZERO, 4, 0.05);
        let p = vsm_page_matrix(face, VsmTreeKind::Cube, 0, 4, 4, 2, 2);
        assert!(
            vsm_page_sees_sphere(&p, Vec3::new(0.6, -0.6, 6.0), 1.0),
            "an infinite far plane rejected a caster in front of the light"
        );
        assert!(
            !vsm_page_sees_sphere(&p, Vec3::new(0.0, 0.0, -6.0), 1.0),
            "a caster BEHIND a cube face was kept"
        );
    }

    /// The six cube faces **tile the sphere**: every direction lands inside
    /// exactly one face's frustum.
    #[test]
    fn the_cube_faces_cover_every_direction_exactly_once() {
        let mats: Vec<Mat4> = (0..6)
            .map(|f| cube_face_matrix(Vec3::ZERO, f, 0.1))
            .collect();
        let mut checked = 0;
        for i in 0..24 {
            for j in 0..12 {
                let theta = std::f32::consts::TAU * (i as f32 + 0.37) / 24.0;
                let phi = std::f32::consts::PI * (j as f32 + 0.41) / 12.0;
                let d =
                    Vec3::new(phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin()) * 5.0;
                let hits = mats
                    .iter()
                    .filter(|m| {
                        let c = **m * d.extend(1.0);
                        c.w > 0.0 && {
                            let n = c.truncate() / c.w;
                            n.x.abs() <= 1.0 && n.y.abs() <= 1.0 && n.z > 0.0 && n.z <= 1.0
                        }
                    })
                    .count();
                assert_eq!(hits, 1, "direction {d:?} landed on {hits} faces");
                checked += 1;
            }
        }
        assert_eq!(checked, 288, "the sweep shrank");
    }
}
