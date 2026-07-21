//! 2D rendering data + the CPU sprite batcher (P8.1a).
//!
//! # Seam: `inf-render-2d` ↔ `inf-render`
//!
//! This crate is the **pure-CPU data + batching half** of the 2D pipeline. It
//! owns [`SpriteInstance`], the sort/batch algorithm, and the quad / UV / pivot
//! math (mirrored 1:1 by the vertex shader) — with **no wgpu dependency**.
//!
//! `inf-render` depends on this crate, embeds `Vec<SpriteInstance>` in its
//! `RenderScene`, and owns the **GPU half** (texture cache + sprite render pass)
//! in its `passes::sprite` module.
//!
//! We chose this "batcher/data crate" seam — documented as acceptable in the
//! P8.1a brief — rather than putting the render-graph node here, because the
//! node must be constructed by `inf-render::EngineRenderer`; a node living in
//! `inf-render-2d` would force `inf-render → inf-render-2d → inf-render`, a
//! dependency cycle. Keeping the node in `inf-render` and the batcher here is
//! acyclic and keeps the CPU math independently unit-testable.
//!
//! ## Alpha convention
//! Sprites use **straight (non-premultiplied) alpha**: the fragment returns
//! `texel * tint` unmodified and the pipeline blends with
//! `src=SrcAlpha, dst=OneMinusSrcAlpha`. See `inf-render`'s sprite pass.

use glam::{DVec3, Vec2, Vec3};

pub mod nine_slice;
pub mod text;
pub mod tilemap;
pub use nine_slice::{expand_nine_slice, NineSliceParams};
pub use text::{
    builtin_font_rgba8, expand_text, HAlign, TextParams, BUILTIN_FONT_COLS, BUILTIN_FONT_FIRST_CP,
    BUILTIN_FONT_ROWS,
};
pub use tilemap::{
    aabb_visible, atlas_uv, chunk_world_aabb, expand_chunk, RenderChunk, RenderTilemap,
    TilemapParams, TILE_CHUNK_DIM,
};

/// Opaque GPU texture identity (a hash of an asset GUID, see
/// [`handle_from_guid`]). `0` is reserved to mean "no texture" → the renderer's
/// 1×1 white fallback.
pub type TextureHandle = u64;

/// The reserved handle meaning "no texture / use the 1×1 white fallback".
pub const WHITE_TEXTURE: TextureHandle = 0;

/// Billboard mode packed into a [`SpriteInstance`] (mirrors
/// `inf_ecs::BillboardMode` — kept as a `u8` here so this pure-CPU crate stays
/// free of an `inf-ecs` dependency). `0` = planar (the pre-2.5D behaviour),
/// `1` = spherical (full camera-facing), `2` = cylindrical (upright about world
/// +Y). The value rides in `SpriteRaw.flags.z`; the vertex shader orients the
/// quad by the camera basis when it is non-zero. See `sprite.wgsl`.
pub const BILLBOARD_NONE: u8 = 0;
pub const BILLBOARD_SPHERICAL: u8 = 1;
pub const BILLBOARD_CYLINDRICAL: u8 = 2;

/// The reserved handle for the built-in 8×8 bitmap font atlas
/// ([`builtin_font_rgba8`]). The sprite pass uploads that atlas under this
/// handle at startup, so a `Text2D` with no font asset (`font_texture = None`)
/// resolves here. A distinctive high value that [`handle_from_guid`] is
/// astronomically unlikely to collide with (and a collision would merely alias
/// one texture to the font — harmless).
pub const BUILTIN_FONT_TEXTURE: TextureHandle = 0xF047_0000_0000_0001;

/// One CPU-side sprite ready for batching. World position is f64 (the render
/// pass rebases it against the floating origin, exactly like mesh instances);
/// everything else is render-space f32.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SpriteInstance {
    /// World-space position of the sprite's pivot (f64 — architecture rule 3).
    pub position: DVec3,
    /// Sprite extent in world units (width, height).
    pub size: Vec2,
    /// Normalized anchor in `[0,1]²` — `(0.5, 0.5)` centers the quad on
    /// `position`, `(0,0)` bottom-left, `(1,1)` top-right.
    pub pivot: Vec2,
    /// Rotation about +Z (world), radians, CCW.
    pub rotation: f32,
    /// Atlas sub-rect: minimum UV (top-left of the source region).
    pub uv_min: Vec2,
    /// Atlas sub-rect: maximum UV (bottom-right of the source region).
    pub uv_max: Vec2,
    /// Linear straight-alpha tint (rgba), multiplied with the texel.
    pub color: [f32; 4],
    /// Which GPU texture to sample; [`WHITE_TEXTURE`] = the white fallback.
    pub texture: TextureHandle,
    /// Coarse draw bucket (lower = further back).
    pub sorting_layer: i32,
    /// Fine ordering within a layer (lower = further back).
    pub order: i32,
    pub flip_x: bool,
    pub flip_y: bool,
    /// Camera-facing mode (P8.4a): [`BILLBOARD_NONE`] (planar, the default),
    /// [`BILLBOARD_SPHERICAL`], or [`BILLBOARD_CYLINDRICAL`]. Non-zero orients
    /// the quad by the camera basis in the vertex shader.
    pub billboard: u8,
}

impl Default for SpriteInstance {
    fn default() -> Self {
        Self {
            position: DVec3::ZERO,
            size: Vec2::ONE,
            pivot: Vec2::splat(0.5),
            rotation: 0.0,
            uv_min: Vec2::ZERO,
            uv_max: Vec2::ONE,
            color: [1.0; 4],
            texture: WHITE_TEXTURE,
            sorting_layer: 0,
            order: 0,
            flip_x: false,
            flip_y: false,
            billboard: BILLBOARD_NONE,
        }
    }
}

impl SpriteInstance {
    /// A centered, full-UV, white-tinted sprite at `position`.
    pub fn at(position: DVec3) -> Self {
        Self {
            position,
            ..Default::default()
        }
    }
}

/// The painter-sort key: `(sorting_layer, order, texture)`. Layer/order impose
/// the required draw order; the texture tie-break lets same-position sprites
/// batch by texture (their relative order is undefined anyway, so grouping is
/// free). Sorting with a **stable** sort keeps input order among fully-equal
/// keys — the determinism guarantee.
#[inline]
pub fn sort_key(s: &SpriteInstance) -> (i32, i32, TextureHandle) {
    (s.sorting_layer, s.order, s.texture)
}

/// A contiguous run of sorted sprites sharing one texture → one draw call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpriteBatch {
    pub texture: TextureHandle,
    /// First instance index into [`BatchedSprites::instances`].
    pub start: u32,
    /// Number of instances in this run.
    pub count: u32,
}

/// The batcher output: sprites in final draw order plus per-texture draw runs.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BatchedSprites {
    /// Sprites in draw order (painter-sorted).
    pub instances: Vec<SpriteInstance>,
    /// Draw runs; each is a maximal run of equal-texture consecutive sprites.
    pub batches: Vec<SpriteBatch>,
}

/// A pre-expanded, already-in-draw-order run of sprite instances that bypasses
/// the loose-sprite sort — e.g. one expanded tilemap chunk. Every instance
/// shares one `texture` and one `(sorting_layer, order)`, so the whole run is a
/// single contiguous batch. Used for tilemaps: expanding 100k tiles as loose
/// [`SpriteInstance`]s would dominate the batcher's `O(n log n)` sort, whereas a
/// per-chunk prebatched run is placed in `O(1)` (plus a stable sort over the
/// handful of runs).
#[derive(Clone, Debug, PartialEq)]
pub struct PrebatchedRun {
    pub texture: TextureHandle,
    pub sorting_layer: i32,
    pub order: i32,
    /// Instances in final intra-run draw order (row-major chunk expansion).
    pub instances: Vec<SpriteInstance>,
}

/// Sort `sprites` into draw order and split them into per-texture batches.
///
/// Sorting is **stable** by `(layer, order, texture)`, so:
/// * later layers/orders always draw on top (painter's algorithm), and
/// * sprites that are fully order-equal keep their input order (determinism).
///
/// Batches are maximal runs of consecutive equal-texture sprites in that order,
/// so a texture can legitimately appear in several batches when interleaved by
/// order — the minimum number of state changes the ordering constraint allows.
pub fn batch_sprites(sprites: &[SpriteInstance]) -> BatchedSprites {
    batch_scene(sprites, &[])
}

/// Batch loose `sprites` together with pre-expanded `prebatched` runs (tilemap
/// chunks) into one final draw order.
///
/// **Ordering semantics.** The unified painter order is by `(sorting_layer,
/// order)`. Loose sprites and prebatched runs interleave by that key; within an
/// *equal* `(layer, order)`:
///   1. all loose sprites are placed first (sorted among themselves by texture,
///      then stable by input order — the P8.1a guarantee), then
///   2. the prebatched runs, in the order they were passed (the host passes them
///      in tilemap-entity order, so this is deterministic and controllable).
///
/// After merging, batches are recomputed as maximal equal-texture runs over the
/// final order — so a prebatched run that happens to share a neighbour's texture
/// coalesces into one draw call rather than forcing a redundant state change.
pub fn batch_scene(sprites: &[SpriteInstance], prebatched: &[PrebatchedRun]) -> BatchedSprites {
    // Loose sprites in painter order (stable by (layer, order, texture)).
    let mut loose = sprites.to_vec();
    loose.sort_by_key(sort_key);

    // Prebatched runs ordered by (layer, order); a stable sort preserves the
    // caller's (tilemap-entity) order for equal keys.
    let mut runs: Vec<&PrebatchedRun> = prebatched.iter().collect();
    runs.sort_by_key(|r| (r.sorting_layer, r.order));

    // Merge: walk both sequences, emitting the lower (layer, order) first and,
    // on a tie, all equal-key loose sprites before the prebatched runs.
    let mut instances: Vec<SpriteInstance> =
        Vec::with_capacity(loose.len() + runs.iter().map(|r| r.instances.len()).sum::<usize>());
    let mut li = 0;
    let mut ri = 0;
    while li < loose.len() || ri < runs.len() {
        let loose_key = loose.get(li).map(|s| (s.sorting_layer, s.order));
        let run_key = runs.get(ri).map(|r| (r.sorting_layer, r.order));
        let take_loose = match (loose_key, run_key) {
            (Some(l), Some(r)) => l <= r, // tie → loose first
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };
        if take_loose {
            let key = loose_key.unwrap();
            while li < loose.len() && (loose[li].sorting_layer, loose[li].order) == key {
                instances.push(loose[li]);
                li += 1;
            }
        } else {
            instances.extend_from_slice(&runs[ri].instances);
            ri += 1;
        }
    }

    // Maximal equal-texture runs over the final order.
    let mut batches = Vec::new();
    let mut i = 0;
    while i < instances.len() {
        let texture = instances[i].texture;
        let start = i;
        while i < instances.len() && instances[i].texture == texture {
            i += 1;
        }
        batches.push(SpriteBatch {
            texture,
            start: start as u32,
            count: (i - start) as u32,
        });
    }
    BatchedSprites { instances, batches }
}

/// Unit-quad corner `(x, y)` in `{0,1}²` for a triangle-strip vertex index
/// `0..4`: `0=(0,0) 1=(1,0) 2=(0,1) 3=(1,1)`. **Must** match the `vs` entry in
/// `sprite.wgsl` (`x = vi & 1`, `y = (vi >> 1) & 1`).
#[inline]
pub fn unit_corner(vertex_index: u32) -> Vec2 {
    Vec2::new((vertex_index & 1) as f32, ((vertex_index >> 1) & 1) as f32)
}

/// The pivot-relative, size-scaled, Z-rotated **local plane offset** of corner
/// `vertex_index` (added to the sprite's render-local center in the shader).
/// Mirrors `sprite.wgsl` exactly so tests validate the shipped math.
pub fn corner_offset(s: &SpriteInstance, vertex_index: u32) -> Vec2 {
    let c = unit_corner(vertex_index);
    let local = Vec2::new((c.x - s.pivot.x) * s.size.x, (c.y - s.pivot.y) * s.size.y);
    let (sin, cos) = s.rotation.sin_cos();
    Vec2::new(local.x * cos - local.y * sin, local.x * sin + local.y * cos)
}

/// The camera-facing basis a billboard mode uses to place its quad, given the
/// camera's render-local `right` and `up` vectors. Mirrors the branch in
/// `sprite.wgsl` **exactly** so the orientation is unit-testable:
///
/// * [`BILLBOARD_SPHERICAL`] → `(right, up)` verbatim (full camera-facing).
/// * [`BILLBOARD_CYLINDRICAL`] → world-up `(0,1,0)` and the horizontal component
///   of `right` (renormalized; falls back to `+X` if degenerate), so the card
///   stays upright and only yaws toward the camera.
///
/// [`BILLBOARD_NONE`] never calls this (its quad lies in the world XY plane).
pub fn billboard_basis(mode: u8, cam_right: Vec3, cam_up: Vec3) -> (Vec3, Vec3) {
    match mode {
        BILLBOARD_CYLINDRICAL => {
            let flat = Vec3::new(cam_right.x, 0.0, cam_right.z);
            let right = if flat.length_squared() > 1e-12 {
                flat.normalize()
            } else {
                Vec3::X
            };
            (right, Vec3::Y)
        }
        // Spherical (and any non-zero fallthrough): full camera basis.
        _ => (cam_right, cam_up),
    }
}

/// The world-space offset of corner `vertex_index` for a **billboarded** sprite:
/// the same pivot-relative, size-scaled, Z-rotated local plane offset as
/// [`corner_offset`], but mapped onto the camera-facing `(right, up)` basis from
/// [`billboard_basis`] instead of the world XY plane. Mirrors `sprite.wgsl`.
pub fn corner_offset_billboard(
    s: &SpriteInstance,
    vertex_index: u32,
    cam_right: Vec3,
    cam_up: Vec3,
) -> Vec3 {
    let local = corner_offset(s, vertex_index); // pivot/size/rotation in plane
    let (right, up) = billboard_basis(s.billboard, cam_right, cam_up);
    right * local.x + up * local.y
}

/// The texture UV of corner `vertex_index`, applying the atlas rect and flips.
/// The quad's +Y (top, `corner.y == 1`) maps to `uv_min.y` (texture top), since
/// texture V grows downward. Mirrors `sprite.wgsl` exactly.
pub fn corner_uv(s: &SpriteInstance, vertex_index: u32) -> Vec2 {
    let c = unit_corner(vertex_index);
    let mut u = c.x;
    let mut v = 1.0 - c.y;
    if s.flip_x {
        u = 1.0 - u;
    }
    if s.flip_y {
        v = 1.0 - v;
    }
    Vec2::new(
        s.uv_min.x + (s.uv_max.x - s.uv_min.x) * u,
        s.uv_min.y + (s.uv_max.y - s.uv_min.y) * v,
    )
}

/// Map a 128-bit asset GUID to a non-zero [`TextureHandle`]. The renderer keys
/// its GPU texture cache on this. `0` (the white-fallback sentinel) is never
/// produced.
#[inline]
pub fn handle_from_guid(guid: u128) -> TextureHandle {
    let lo = guid as u64;
    let hi = (guid >> 64) as u64;
    let h = lo ^ hi.rotate_left(32);
    // Never collide with the white-fallback sentinel.
    if h == WHITE_TEXTURE {
        0xffff_ffff_ffff_ffff
    } else {
        h
    }
}

/// The 2D-light radial falloff, mirroring `sprite.wgsl`'s
/// `smoothstep(radius, 0.0, dist)` **exactly**: `1` at the light, smoothly
/// easing to `0` at (and past) `radius`. Kept CPU-side so the shading curve is
/// unit-testable, the same way [`corner_offset`] mirrors the vertex shader. A
/// non-positive radius contributes nothing (matches the shader's guard).
pub fn light2d_falloff(radius: f32, dist: f32) -> f32 {
    if radius <= 0.0 {
        return 0.0;
    }
    // smoothstep(edge0=radius, edge1=0, x=dist): t = clamp((radius-dist)/radius).
    let t = ((radius - dist) / radius).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sprite(layer: i32, order: i32, texture: TextureHandle) -> SpriteInstance {
        SpriteInstance {
            sorting_layer: layer,
            order,
            texture,
            ..Default::default()
        }
    }

    #[test]
    fn sort_orders_by_layer_then_order_then_texture() {
        let input = vec![
            sprite(1, 0, 10),
            sprite(0, 5, 10),
            sprite(0, 0, 20),
            sprite(0, 0, 10),
        ];
        let out = batch_sprites(&input);
        let keys: Vec<_> = out.instances.iter().map(sort_key).collect();
        assert_eq!(keys, vec![(0, 0, 10), (0, 0, 20), (0, 5, 10), (1, 0, 10)]);
    }

    #[test]
    fn sort_is_stable_for_equal_keys() {
        // Three sprites with an identical key but distinguishable positions:
        // a stable sort must preserve their input order.
        let mut a = sprite(0, 0, 7);
        a.position = DVec3::new(1.0, 0.0, 0.0);
        let mut b = sprite(0, 0, 7);
        b.position = DVec3::new(2.0, 0.0, 0.0);
        let mut c = sprite(0, 0, 7);
        c.position = DVec3::new(3.0, 0.0, 0.0);
        let out = batch_sprites(&[a, b, c]);
        let xs: Vec<f64> = out.instances.iter().map(|s| s.position.x).collect();
        assert_eq!(xs, vec![1.0, 2.0, 3.0]);
        // One batch (all share the texture).
        assert_eq!(out.batches.len(), 1);
        assert_eq!(out.batches[0].count, 3);
    }

    #[test]
    fn batches_split_per_texture_run() {
        // Same layer/order, three textures interleaved after the sort.
        let out = batch_sprites(&[
            sprite(0, 0, 30),
            sprite(0, 0, 10),
            sprite(0, 0, 20),
            sprite(0, 0, 10),
        ]);
        // Sorted by texture within the equal (layer, order): 10,10,20,30.
        let batches = &out.batches;
        assert_eq!(batches.len(), 3);
        assert_eq!(
            batches[0],
            SpriteBatch {
                texture: 10,
                start: 0,
                count: 2
            }
        );
        assert_eq!(
            batches[1],
            SpriteBatch {
                texture: 20,
                start: 2,
                count: 1
            }
        );
        assert_eq!(
            batches[2],
            SpriteBatch {
                texture: 30,
                start: 3,
                count: 1
            }
        );
    }

    #[test]
    fn texture_can_reappear_when_order_forces_it() {
        // T10 at order 0, T20 at order 1, T10 at order 2 — the middle sprite
        // forbids merging the two T10 draws.
        let out = batch_sprites(&[sprite(0, 0, 10), sprite(0, 2, 10), sprite(0, 1, 20)]);
        let tex: Vec<_> = out.batches.iter().map(|b| (b.texture, b.count)).collect();
        assert_eq!(tex, vec![(10, 1), (20, 1), (10, 1)]);
    }

    fn run(layer: i32, order: i32, texture: TextureHandle, n: usize) -> PrebatchedRun {
        let mut s = sprite(layer, order, texture);
        // Tag each instance's position so we can track run identity/order.
        let instances = (0..n)
            .map(|k| {
                s.position = DVec3::new(texture as f64, k as f64, 0.0);
                s
            })
            .collect();
        PrebatchedRun {
            texture,
            sorting_layer: layer,
            order,
            instances,
        }
    }

    #[test]
    fn batch_scene_places_loose_before_prebatched_on_equal_key() {
        // One loose sprite and one prebatched run at the SAME (layer, order):
        // the loose sprite draws first, the run after.
        let loose = vec![sprite(0, 0, 10)];
        let runs = vec![run(0, 0, 20, 2)];
        let out = batch_scene(&loose, &runs);
        let tex: Vec<_> = out.instances.iter().map(|s| s.texture).collect();
        assert_eq!(tex, vec![10, 20, 20]);
        // Two batches: loose(10), then the run(20).
        assert_eq!(
            out.batches,
            vec![
                SpriteBatch {
                    texture: 10,
                    start: 0,
                    count: 1
                },
                SpriteBatch {
                    texture: 20,
                    start: 1,
                    count: 2
                },
            ]
        );
    }

    #[test]
    fn batch_scene_interleaves_runs_and_loose_by_layer() {
        // Loose sprite on layer 1; two runs on layers 0 and 2. Painter order must
        // be run(0) → loose(1) → run(2).
        let loose = vec![sprite(1, 0, 50)];
        let runs = vec![run(2, 0, 70, 1), run(0, 0, 60, 1)];
        let out = batch_scene(&loose, &runs);
        let tex: Vec<_> = out.instances.iter().map(|s| s.texture).collect();
        assert_eq!(tex, vec![60, 50, 70]);
    }

    #[test]
    fn batch_scene_preserves_run_input_order_on_equal_key() {
        // Two runs with the same (layer, order): they keep the order passed in
        // (the host's tilemap-entity order) — determinism.
        let runs = vec![run(0, 0, 80, 1), run(0, 0, 81, 1)];
        let out = batch_scene(&[], &runs);
        let tex: Vec<_> = out.instances.iter().map(|s| s.texture).collect();
        assert_eq!(tex, vec![80, 81]);
        // Reversing the input reverses the draw order (order is caller-controlled).
        let out2 = batch_scene(&[], &[runs[1].clone(), runs[0].clone()]);
        let tex2: Vec<_> = out2.instances.iter().map(|s| s.texture).collect();
        assert_eq!(tex2, vec![81, 80]);
    }

    #[test]
    fn batch_scene_matches_batch_sprites_with_no_runs() {
        let loose = vec![sprite(1, 0, 10), sprite(0, 0, 20), sprite(0, 0, 10)];
        assert_eq!(batch_scene(&loose, &[]), batch_sprites(&loose));
    }

    #[test]
    fn pivot_centers_and_corners_span_size() {
        let s = SpriteInstance {
            size: Vec2::new(4.0, 2.0),
            pivot: Vec2::splat(0.5),
            ..Default::default()
        };
        // Centered pivot: bottom-left corner is (-w/2, -h/2), top-right (+w/2,+h/2).
        assert_eq!(corner_offset(&s, 0), Vec2::new(-2.0, -1.0));
        assert_eq!(corner_offset(&s, 3), Vec2::new(2.0, 1.0));
        // Bottom-left pivot puts the origin at the sprite's bottom-left corner.
        let bl = SpriteInstance {
            pivot: Vec2::ZERO,
            ..s
        };
        assert_eq!(corner_offset(&bl, 0), Vec2::new(0.0, 0.0));
        assert_eq!(corner_offset(&bl, 3), Vec2::new(4.0, 2.0));
    }

    #[test]
    fn rotation_ninety_degrees_maps_x_to_y() {
        let s = SpriteInstance {
            size: Vec2::new(2.0, 2.0),
            pivot: Vec2::ZERO,
            rotation: std::f32::consts::FRAC_PI_2,
            ..Default::default()
        };
        // Corner (1,0)*size = (2,0) rotated +90° → (0, 2).
        let o = corner_offset(&s, 1);
        assert!((o - Vec2::new(0.0, 2.0)).length() < 1e-5, "{o:?}");
    }

    #[test]
    fn uv_rect_maps_corners_and_flips() {
        let s = SpriteInstance {
            uv_min: Vec2::new(0.25, 0.5),
            uv_max: Vec2::new(0.75, 1.0),
            ..Default::default()
        };
        // Top-left quad corner (0,1) → texture top-left (uv_min).
        assert_eq!(corner_uv(&s, 2), Vec2::new(0.25, 0.5));
        // Bottom-right quad corner (1,0) → texture bottom-right (uv_max).
        assert_eq!(corner_uv(&s, 1), Vec2::new(0.75, 1.0));

        // flip_x swaps left/right; flip_y swaps top/bottom.
        let fx = SpriteInstance { flip_x: true, ..s };
        assert_eq!(corner_uv(&fx, 2), Vec2::new(0.75, 0.5));
        let fy = SpriteInstance { flip_y: true, ..s };
        assert_eq!(corner_uv(&fy, 2), Vec2::new(0.25, 1.0));
    }

    #[test]
    fn billboard_basis_spherical_is_camera_basis_cylindrical_is_upright() {
        // A camera looking down -Z with up +Y: right = +X.
        let right = Vec3::X;
        let up = Vec3::Y;
        assert_eq!(billboard_basis(BILLBOARD_SPHERICAL, right, up), (right, up));

        // Tilt the camera so its "up" gains a Z component (pitched down): the
        // cylindrical basis keeps world up and flattens right to horizontal.
        let tilted_up = Vec3::new(0.0, 0.7, 0.7).normalize();
        let tilted_right = Vec3::X; // still horizontal
        let (r, u) = billboard_basis(BILLBOARD_CYLINDRICAL, tilted_right, tilted_up);
        assert_eq!(u, Vec3::Y, "cylindrical stays upright about world Y");
        assert!(
            (r - Vec3::X).length() < 1e-6,
            "right flattened to horizontal"
        );

        // A camera rolled so its right has a Y component: cylindrical drops the Y.
        let rolled_right = Vec3::new(0.6, 0.8, 0.0).normalize();
        let (r2, _) = billboard_basis(BILLBOARD_CYLINDRICAL, rolled_right, Vec3::Y);
        assert!(r2.y.abs() < 1e-6, "cylindrical right is horizontal: {r2:?}");
    }

    #[test]
    fn corner_offset_billboard_maps_plane_onto_camera_basis() {
        // Spherical billboard facing a -Z camera: the world XY plane offset is
        // reproduced (right=+X, up=+Y), so it matches the planar offset embedded
        // in Z=0.
        let s = SpriteInstance {
            size: Vec2::new(4.0, 2.0),
            pivot: Vec2::splat(0.5),
            billboard: BILLBOARD_SPHERICAL,
            ..Default::default()
        };
        let o = corner_offset_billboard(&s, 3, Vec3::X, Vec3::Y);
        assert!((o - Vec3::new(2.0, 1.0, 0.0)).length() < 1e-6, "{o:?}");
        // With the camera pointing down +Z (right=-X), the card flips to face it.
        let o2 = corner_offset_billboard(&s, 3, -Vec3::X, Vec3::Y);
        assert!((o2 - Vec3::new(-2.0, 1.0, 0.0)).length() < 1e-6, "{o2:?}");
    }

    #[test]
    fn handle_from_guid_is_never_the_white_sentinel() {
        assert_ne!(handle_from_guid(0), WHITE_TEXTURE);
        // Distinct GUIDs generally map to distinct handles.
        assert_ne!(handle_from_guid(1), handle_from_guid(2));
    }

    #[test]
    fn light2d_falloff_is_one_at_center_zero_at_radius() {
        assert_eq!(light2d_falloff(2.0, 0.0), 1.0);
        assert_eq!(light2d_falloff(2.0, 2.0), 0.0);
        // Past the radius clamps to zero (never negative).
        assert_eq!(light2d_falloff(2.0, 5.0), 0.0);
        // Smooth + strictly decreasing across the interval.
        let mid = light2d_falloff(2.0, 1.0);
        assert!(mid > 0.0 && mid < 1.0);
        assert!(light2d_falloff(2.0, 0.5) > light2d_falloff(2.0, 1.5));
        // Degenerate radius contributes nothing (matches the shader guard).
        assert_eq!(light2d_falloff(0.0, 0.0), 0.0);
    }
}
