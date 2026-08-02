//! Built-in primitive meshes (R-P1): per-kind CPU geometry generators + a single
//! packed GPU buffer pair the mesh-drawing passes share.
//!
//! Every scene primitive used to render as a cube; [`PrimMesh`] adds Sphere /
//! Plane / Cylinder / Cone as real geometry. The generators are **pure** (no GPU)
//! and produce `MeshVertex { pos, normal }` — the exact layout every mesh pass
//! already binds — so a pass swaps its private cube buffer for one [`PrimGpu`]
//! and issues up to five `draw_indexed` calls (one per kind that has instances).
//!
//! Convention (matches the cube): unit shapes centred on the origin, extent
//! `±0.5`; all indices wind **CCW when seen from outside** (front faces), since
//! the pipelines cull `Face::Back`. All trig goes through [`inf_math::psin64`] /
//! [`inf_math::pcos64`] (accurate **and** bit-portable) so the committed vertex
//! bytes — and therefore the goldens — are identical on every platform (house
//! law: no `std` trig in deterministic vertex data).

use inf_math::{pcos64, psin64};

use crate::gpu::GpuContext;
use crate::passes::mesh::MeshVertex;

/// A built-in primitive-mesh kind. `#[default]` is `Cube` so any `MeshInstance`
/// that doesn't set a kind draws exactly as before (byte-stable pre-R-P1 scenes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum PrimMesh {
    #[default]
    Cube,
    Sphere,
    Plane,
    Cylinder,
    Cone,
}

impl PrimMesh {
    /// All kinds in the **canonical packing order** — `Cube` first (index 0) so an
    /// all-cube scene buckets into slot 0 and stays byte-identical to the old
    /// linear pack. Bucketing and the packed GPU buffer both use this order.
    pub const ALL: [PrimMesh; 5] = [
        PrimMesh::Cube,
        PrimMesh::Sphere,
        PrimMesh::Plane,
        PrimMesh::Cylinder,
        PrimMesh::Cone,
    ];

    /// Dense index into [`ALL`](Self::ALL) / the per-kind range arrays.
    #[inline]
    pub const fn index(self) -> usize {
        match self {
            PrimMesh::Cube => 0,
            PrimMesh::Sphere => 1,
            PrimMesh::Plane => 2,
            PrimMesh::Cylinder => 3,
            PrimMesh::Cone => 4,
        }
    }

    /// The radius of this kind's bounding sphere at **unit scale**, centred at the
    /// origin (every built-in primitive is authored centred).
    ///
    /// Exact, not measured: the cube's half-diagonal is `√3/2`, the plane's is
    /// `√2/2` (it lies flat in XZ), the cylinder's and the cone's are the
    /// half-diagonal of a `0.5`-radius, `1`-tall bound — `√2/2` — and the sphere's
    /// is its own `0.5`. Multiplied by an instance's uniform scale it gives the
    /// conservative cull sphere the P18.5 scatter compute tests, and a
    /// conservative-by-construction bound is what lets that cull be *subtractive*
    /// rather than approximate. Pinned against the actual geometry by
    /// `bounding_radius_bounds_every_vertex`.
    pub const fn bounding_radius(self) -> f32 {
        // `sqrt` is not const, so the two irrational half-diagonals come from
        // `std::f32::consts` (√2/2 is `FRAC_1_SQRT_2`) and one folded literal
        // (√3/2), rather than from three hand-typed decimals clippy would rightly
        // flag as approximating a named constant.
        const HALF_SQRT_3: f32 = 0.866_025_4;
        match self {
            PrimMesh::Cube => HALF_SQRT_3,
            PrimMesh::Sphere => 0.5,
            PrimMesh::Plane => std::f32::consts::FRAC_1_SQRT_2,
            PrimMesh::Cylinder => std::f32::consts::FRAC_1_SQRT_2,
            PrimMesh::Cone => std::f32::consts::FRAC_1_SQRT_2,
        }
    }

    /// This kind's CPU geometry (`pos + normal`, CCW-outside).
    pub fn geometry(self) -> (Vec<MeshVertex>, Vec<u16>) {
        match self {
            PrimMesh::Cube => cube_geometry(),
            PrimMesh::Sphere => sphere_geometry(),
            PrimMesh::Plane => plane_geometry(),
            PrimMesh::Cylinder => cylinder_geometry(),
            PrimMesh::Cone => cone_geometry(),
        }
    }
}

/// Where one kind's geometry lives inside the packed [`PrimGpu`] buffers: an index
/// sub-range plus the `base_vertex` its indices are relative to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrimRange {
    pub index_start: u32,
    pub index_count: u32,
    pub base_vertex: i32,
}

// ── Tuning ───────────────────────────────────────────────────────────────────

const SPHERE_STACKS: u32 = 16;
const SPHERE_SLICES: u32 = 24;
const RADIAL_SEGMENTS: u32 = 24; // cylinder + cone
const TAU64: f64 = std::f64::consts::TAU;
const PI64: f64 = std::f64::consts::PI;

#[inline]
fn v(pos: [f32; 3], normal: [f32; 3]) -> MeshVertex {
    MeshVertex { pos, normal }
}

/// Unit cube centred at the origin (extent `±0.5`), 24 verts / 36 indices. Moved
/// here from `passes::mesh` (re-exported there for existing references); the byte
/// layout is unchanged so every cube golden stays identical.
pub fn cube_geometry() -> (Vec<MeshVertex>, Vec<u16>) {
    let faces: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
        // (normal, tangent u, tangent v)
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    ];
    let mut verts = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (n, u, w) in faces {
        let n3 = glam::Vec3::from(n);
        let u3 = glam::Vec3::from(u);
        let v3 = glam::Vec3::from(w);
        let base = verts.len() as u16;
        for (su, sv) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            let p = (n3 + u3 * su + v3 * sv) * 0.5;
            verts.push(v(p.to_array(), n));
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (verts, indices)
}

/// UV sphere, radius `0.5`, [`SPHERE_STACKS`] × [`SPHERE_SLICES`]. Ported from the
/// thumbnailer's `unit_sphere` (stripped to pos+normal, scaled to `±0.5`) with the
/// winding flipped to CCW-outside for the viewport's back-face cull. Normals equal
/// the outward position direction. Trig via `psin64`/`pcos64` (bit-portable).
pub fn sphere_geometry() -> (Vec<MeshVertex>, Vec<u16>) {
    let (stacks, slices) = (SPHERE_STACKS, SPHERE_SLICES);
    let mut verts = Vec::with_capacity(((stacks + 1) * (slices + 1)) as usize);
    for i in 0..=stacks {
        let phi = PI64 * i as f64 / stacks as f64;
        let (sp, cp) = (psin64(phi), pcos64(phi));
        for j in 0..=slices {
            let theta = TAU64 * j as f64 / slices as f64;
            let (st, ct) = (psin64(theta), pcos64(theta));
            let n = [(sp * ct) as f32, cp as f32, (sp * st) as f32];
            verts.push(v([n[0] * 0.5, n[1] * 0.5, n[2] * 0.5], n));
        }
    }
    let row = slices + 1;
    let mut indices = Vec::with_capacity((stacks * slices * 6) as usize);
    for i in 0..stacks {
        for j in 0..slices {
            let a = (i * row + j) as u16;
            let b = a + row as u16;
            // CCW seen from outside (opposite the thumbnailer's inward winding).
            indices.extend_from_slice(&[a, a + 1, b, a + 1, b + 1, b]);
        }
    }
    (verts, indices)
}

/// Unit plane, `1×1` in the XZ plane at `y = 0`, single-sided facing `+Y`. Winding
/// mirrors the cube's top face (CCW seen from above).
pub fn plane_geometry() -> (Vec<MeshVertex>, Vec<u16>) {
    let n = [0.0, 1.0, 0.0];
    let verts = vec![
        v([-0.5, 0.0, 0.5], n),
        v([0.5, 0.0, 0.5], n),
        v([0.5, 0.0, -0.5], n),
        v([-0.5, 0.0, -0.5], n),
    ];
    let indices = vec![0u16, 1, 2, 0, 2, 3];
    (verts, indices)
}

/// Unit cylinder: radius `0.5`, height `1` (`y ∈ [-0.5, 0.5]`),
/// [`RADIAL_SEGMENTS`] around, both caps closed. Radial side normals; `±Y` cap
/// normals. Side winding matches the skinned-cylinder golden (CCW-outside).
pub fn cylinder_geometry() -> (Vec<MeshVertex>, Vec<u16>) {
    let segs = RADIAL_SEGMENTS;
    let mut verts = Vec::new();
    let mut indices = Vec::new();

    // Side rings: bottom [0..segs), top [segs..2segs). Radial normals, wrap-shared.
    let ring: Vec<(f32, f32)> = (0..segs)
        .map(|s| {
            let a = TAU64 * s as f64 / segs as f64;
            (pcos64(a) as f32, psin64(a) as f32)
        })
        .collect();
    for &(c, s) in &ring {
        verts.push(v([0.5 * c, -0.5, 0.5 * s], [c, 0.0, s]));
    }
    for &(c, s) in &ring {
        verts.push(v([0.5 * c, 0.5, 0.5 * s], [c, 0.0, s]));
    }
    for s in 0..segs {
        let s1 = (s + 1) % segs;
        let (bc, bn, tc, tn) = (s as u16, s1 as u16, (segs + s) as u16, (segs + s1) as u16);
        indices.extend_from_slice(&[bc, tc, bn, bn, tc, tn]);
    }

    // Top cap (+Y): centre + rim fan.
    let top_center = verts.len() as u16;
    verts.push(v([0.0, 0.5, 0.0], [0.0, 1.0, 0.0]));
    let top_rim = verts.len() as u16;
    for &(c, s) in &ring {
        verts.push(v([0.5 * c, 0.5, 0.5 * s], [0.0, 1.0, 0.0]));
    }
    for s in 0..segs {
        let s1 = (s + 1) % segs;
        indices.extend_from_slice(&[top_center, top_rim + s1 as u16, top_rim + s as u16]);
    }

    // Bottom cap (−Y): centre + rim fan (reversed winding for the −Y normal).
    let bot_center = verts.len() as u16;
    verts.push(v([0.0, -0.5, 0.0], [0.0, -1.0, 0.0]));
    let bot_rim = verts.len() as u16;
    for &(c, s) in &ring {
        verts.push(v([0.5 * c, -0.5, 0.5 * s], [0.0, -1.0, 0.0]));
    }
    for s in 0..segs {
        let s1 = (s + 1) % segs;
        indices.extend_from_slice(&[bot_center, bot_rim + s as u16, bot_rim + s1 as u16]);
    }

    (verts, indices)
}

/// Unit cone: base radius `0.5` at `y = -0.5`, apex at `y = +0.5`,
/// [`RADIAL_SEGMENTS`] around, base cap closed. Slant side normals
/// (`normalize(cosθ, 0.5, sinθ)`); `−Y` base-cap normal.
pub fn cone_geometry() -> (Vec<MeshVertex>, Vec<u16>) {
    let segs = RADIAL_SEGMENTS;
    let mut verts = Vec::new();
    let mut indices = Vec::new();

    let angle = |s: u32| TAU64 * s as f64 / segs as f64;
    let slant = |a: f64| glam::Vec3::new(pcos64(a) as f32, 0.5, psin64(a) as f32).normalize();

    // Side: one flat triangle per segment (apex, rim[s+1], rim[s] → CCW-outside).
    for s in 0..segs {
        let (a0, a1) = (angle(s), angle(s + 1));
        let am = 0.5 * (a0 + a1);
        let n_apex = slant(am).to_array();
        let n0 = slant(a0).to_array();
        let n1 = slant(a1).to_array();
        let r0 = [0.5 * pcos64(a0) as f32, -0.5, 0.5 * psin64(a0) as f32];
        let r1 = [0.5 * pcos64(a1) as f32, -0.5, 0.5 * psin64(a1) as f32];
        let base = verts.len() as u16;
        verts.push(v([0.0, 0.5, 0.0], n_apex)); // apex
        verts.push(v(r1, n1)); // rim[s+1]
        verts.push(v(r0, n0)); // rim[s]
        indices.extend_from_slice(&[base, base + 1, base + 2]);
    }

    // Base cap (−Y): centre + rim fan.
    let center = verts.len() as u16;
    verts.push(v([0.0, -0.5, 0.0], [0.0, -1.0, 0.0]));
    let rim = verts.len() as u16;
    for s in 0..segs {
        let a = angle(s);
        verts.push(v(
            [0.5 * pcos64(a) as f32, -0.5, 0.5 * psin64(a) as f32],
            [0.0, -1.0, 0.0],
        ));
    }
    for s in 0..segs {
        let s1 = (s + 1) % segs;
        indices.extend_from_slice(&[center, rim + s as u16, rim + s1 as u16]);
    }

    (verts, indices)
}

/// All five kinds concatenated into one vertex list + one `u16` index list, with a
/// [`PrimRange`] per kind (in [`PrimMesh::ALL`] order). Cube is first, so its
/// `base_vertex` is 0 and its index range is `0..36` — identical to the old
/// single-cube draw, which keeps every all-cube golden byte-stable.
pub fn packed_geometry() -> (Vec<MeshVertex>, Vec<u16>, [PrimRange; 5]) {
    let mut verts: Vec<MeshVertex> = Vec::new();
    let mut indices: Vec<u16> = Vec::new();
    let ranges = std::array::from_fn(|k| {
        let (kv, ki) = PrimMesh::ALL[k].geometry();
        let range = PrimRange {
            index_start: indices.len() as u32,
            index_count: ki.len() as u32,
            base_vertex: verts.len() as i32,
        };
        verts.extend(kv);
        indices.extend(ki);
        range
    });
    assert!(
        verts.len() < 65_536,
        "packed primitive vertices {} exceed the u16 index space",
        verts.len()
    );
    (verts, indices, ranges)
}

/// The shared GPU geometry for every built-in primitive: one vertex buffer + one
/// `u16` index buffer holding all five kinds, plus their [`PrimRange`]s. Each mesh
/// pass owns one of these in place of its old private cube buffer and draws each
/// kind's instances with `draw_indexed(range, base_vertex, instance_range)`.
pub struct PrimGpu {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    ranges: [PrimRange; 5],
}

impl PrimGpu {
    pub fn new(gpu: &GpuContext, label: &str) -> Self {
        let (verts, idx, ranges) = packed_geometry();
        let vertices = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}-prim-vertices")),
            size: std::mem::size_of_val(verts.as_slice()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&vertices, 0, bytemuck::cast_slice(&verts));
        let indices = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}-prim-indices")),
            size: std::mem::size_of_val(idx.as_slice()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        gpu.queue
            .write_buffer(&indices, 0, bytemuck::cast_slice(&idx));
        Self {
            vertices,
            indices,
            ranges,
        }
    }

    /// Bind the shared geometry + `instances`, then issue one `draw_indexed` per
    /// kind with a non-empty instance range. `inst_ranges[k]` is the slice of the
    /// (bucket-packed) instance buffer holding kind `k`'s instances — see
    /// `passes::mesh::pack_bucketed`.
    pub fn draw<'p>(
        &'p self,
        pass: &mut wgpu::RenderPass<'p>,
        instances: &'p wgpu::Buffer,
        inst_ranges: &[std::ops::Range<u32>; 5],
    ) {
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        pass.set_vertex_buffer(1, instances.slice(..));
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint16);
        for (k, inst) in inst_ranges.iter().enumerate() {
            if inst.start >= inst.end {
                continue;
            }
            let r = &self.ranges[k];
            pass.draw_indexed(
                r.index_start..r.index_start + r.index_count,
                r.base_vertex,
                inst.clone(),
            );
        }
    }

    /// Draw `kinds.len()` instances **in explicit order**, one `draw_indexed` per
    /// instance, each with its own primitive kind. Used by the translucent pass
    /// (R-P5), whose back-to-front sort mixes kinds so per-kind batching (see
    /// [`draw`](Self::draw)) can't apply — `kinds[i]` is the [`PrimMesh::index`]
    /// of the instance packed at slot `i` of `instances`. Translucent counts are
    /// small, so the per-instance draw is acceptable (documented).
    pub fn draw_sorted<'p>(
        &'p self,
        pass: &mut wgpu::RenderPass<'p>,
        instances: &'p wgpu::Buffer,
        kinds: &[usize],
    ) {
        if kinds.is_empty() {
            return;
        }
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        pass.set_vertex_buffer(1, instances.slice(..));
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint16);
        for (i, &k) in kinds.iter().enumerate() {
            let r = &self.ranges[k];
            let inst = i as u32;
            pass.draw_indexed(
                r.index_start..r.index_start + r.index_count,
                r.base_vertex,
                inst..inst + 1,
            );
        }
    }
}

/// The same five primitives as [`PrimGpu`], uploaded as **storage** buffers for
/// vertex-pulling passes (P18.5 scatter).
///
/// Two buffers, both read in the vertex stage: `vertices` is a flat `array<f32>`
/// with six floats per vertex (position then normal, exactly `MeshVertex`'s
/// layout), and `indices` widens the packed `u16` index list to `u32` because
/// WGSL has no 16-bit scalar type. Widening costs ~4 KB for the whole primitive
/// set and buys an index read that is one array subscript.
///
/// It exists rather than reusing [`PrimGpu`]'s buffers because a wgpu buffer's
/// usage flags are fixed at creation and `VERTEX | INDEX` is not `STORAGE`; the
/// geometry itself is byte-identical, produced by the same [`packed_geometry`].
pub struct PrimStorage {
    pub vertices: wgpu::Buffer,
    pub indices: wgpu::Buffer,
    pub ranges: [PrimRange; 5],
}

impl PrimStorage {
    pub fn new(gpu: &GpuContext, label: &str) -> Self {
        let (verts, idx, ranges) = packed_geometry();
        let mut flat: Vec<f32> = Vec::with_capacity(verts.len() * 6);
        for v in &verts {
            flat.extend_from_slice(&v.pos);
            flat.extend_from_slice(&v.normal);
        }
        let wide: Vec<u32> = idx.iter().map(|i| *i as u32).collect();
        let mk = |name: &str, bytes: &[u8]| {
            let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(name),
                size: bytes.len() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            gpu.queue.write_buffer(&buf, 0, bytes);
            buf
        };
        Self {
            vertices: mk(
                &format!("{label}-prim-storage-vertices"),
                bytemuck::cast_slice(&flat),
            ),
            indices: mk(
                &format!("{label}-prim-storage-indices"),
                bytemuck::cast_slice(&wide),
            ),
            ranges,
        }
    }

    /// The packed range of one kind's geometry.
    pub fn range(&self, mesh: PrimMesh) -> PrimRange {
        self.ranges[mesh.index()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    /// 6·signed volume of a closed triangle mesh; positive ⇒ CCW-outside winding.
    fn six_signed_volume(verts: &[MeshVertex], indices: &[u16]) -> f32 {
        indices
            .chunks(3)
            .map(|t| {
                let a = Vec3::from(verts[t[0] as usize].pos);
                let b = Vec3::from(verts[t[1] as usize].pos);
                let c = Vec3::from(verts[t[2] as usize].pos);
                a.dot(b.cross(c))
            })
            .sum()
    }

    /// The P18.5 scatter cull tests ONE sphere per instance, sized by
    /// [`PrimMesh::bounding_radius`], and the HZB proof needs that bound to be an
    /// over-approximation — a radius one ULP short of a vertex would let the cull
    /// prove a meshlet invisible that is not. So the constant is checked against
    /// the geometry it claims to bound, kind by kind, rather than trusted.
    #[test]
    fn bounding_radius_bounds_every_vertex() {
        for kind in PrimMesh::ALL {
            let (verts, _) = kind.geometry();
            let r = kind.bounding_radius();
            let far = verts
                .iter()
                .map(|v| Vec3::from(v.pos).length())
                .fold(0.0f32, f32::max);
            assert!(
                far <= r,
                "{kind:?}: a vertex at {far} escapes its {r} bounding radius"
            );
            // …and it is TIGHT: a radius that over-bounds by a lot would cull
            // nothing and quietly make the whole test vacuous.
            assert!(
                far >= r - 1e-3,
                "{kind:?}: bounding radius {r} is loose (farthest vertex {far})"
            );
        }
    }

    fn assert_within_half(verts: &[MeshVertex]) {
        for vert in verts {
            let p = Vec3::from(vert.pos);
            assert!(
                p.x.abs() <= 0.5001 && p.y.abs() <= 0.5001 && p.z.abs() <= 0.5001,
                "vertex {p:?} outside ±0.5"
            );
        }
    }

    #[test]
    fn cube_counts_and_bounds() {
        let (vx, ix) = cube_geometry();
        assert_eq!(vx.len(), 24);
        assert_eq!(ix.len(), 36);
        assert_within_half(&vx);
        assert!(six_signed_volume(&vx, &ix) > 0.0);
    }

    #[test]
    fn sphere_counts_winding_bounds() {
        let (vx, ix) = sphere_geometry();
        assert_eq!(
            vx.len(),
            ((SPHERE_STACKS + 1) * (SPHERE_SLICES + 1)) as usize
        );
        assert_eq!(ix.len(), (SPHERE_STACKS * SPHERE_SLICES * 6) as usize);
        assert_within_half(&vx);
        // Closed sphere, CCW-outside.
        assert!(six_signed_volume(&vx, &ix) > 0.0);
        // Radius ≈ 0.5 everywhere (accurate trig, not the demo-grade f32 sine).
        for vert in &vx {
            let r = Vec3::from(vert.pos).length();
            assert!((r - 0.5).abs() < 1e-3, "sphere radius {r}");
        }
    }

    #[test]
    fn plane_is_flat_up_facing_ccw() {
        let (vx, ix) = plane_geometry();
        assert_eq!(vx.len(), 4);
        assert_eq!(ix.len(), 6);
        assert_within_half(&vx);
        for vert in &vx {
            assert_eq!(vert.pos[1], 0.0);
            assert_eq!(vert.normal, [0.0, 1.0, 0.0]);
        }
        // Both triangles wind CCW seen from +Y (geometric normal points +Y).
        for t in ix.chunks(3) {
            let a = Vec3::from(vx[t[0] as usize].pos);
            let b = Vec3::from(vx[t[1] as usize].pos);
            let c = Vec3::from(vx[t[2] as usize].pos);
            let n = (b - a).cross(c - a);
            assert!(n.y > 0.0, "plane triangle not +Y facing: {n:?}");
        }
    }

    #[test]
    fn cylinder_closed_ccw_bounds() {
        let (vx, ix) = cylinder_geometry();
        assert_within_half(&vx);
        assert!(six_signed_volume(&vx, &ix) > 0.0);
    }

    #[test]
    fn cone_closed_ccw_bounds() {
        let (vx, ix) = cone_geometry();
        assert_within_half(&vx);
        assert!(six_signed_volume(&vx, &ix) > 0.0);
    }

    #[test]
    fn packed_ranges_are_contiguous_and_cube_first() {
        let (verts, indices, ranges) = packed_geometry();
        // Cube occupies the head of both buffers (base_vertex 0, indices 0..36).
        assert_eq!(ranges[0].base_vertex, 0);
        assert_eq!(ranges[0].index_start, 0);
        assert_eq!(ranges[0].index_count, 36);
        // Ranges tile the index buffer in order with no gaps/overlaps.
        let mut idx_cursor = 0u32;
        let mut vtx_cursor = 0i32;
        for (k, r) in ranges.iter().enumerate() {
            let (kv, ki) = PrimMesh::ALL[k].geometry();
            assert_eq!(r.index_start, idx_cursor);
            assert_eq!(r.index_count, ki.len() as u32);
            assert_eq!(r.base_vertex, vtx_cursor);
            idx_cursor += ki.len() as u32;
            vtx_cursor += kv.len() as i32;
        }
        assert_eq!(idx_cursor as usize, indices.len());
        assert_eq!(vtx_cursor as usize, verts.len());
        assert!(verts.len() < 65_536);
    }
}
