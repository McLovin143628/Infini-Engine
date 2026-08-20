//! Built-in primitive meshes (R-P1): per-kind CPU geometry generators + a single
//! packed GPU buffer pair the mesh-drawing passes share.
//!
//! Every scene primitive used to render as a cube; [`PrimMesh`] adds Sphere /
//! Plane / Cylinder / Cone as real geometry. The generators are **pure** (no GPU)
//! and produce `MeshVertex { pos, normal, uv }` — the exact layout every mesh
//! pass already binds — so a pass swaps its private cube buffer for one
//! [`PrimGpu`] and issues up to five `draw_indexed` calls (one per kind that has
//! instances).
//!
//! # The five parametrizations (P26.5)
//!
//! The `uv` field arrived with P26.5, and it is what makes `vt_box_uv` — the
//! P26.3 dominant-axis box projection these shapes sampled with — retirable.
//! Each generator now carries a **named** parametrization, because a uv is a
//! visual decision rather than a derivation and the next person is entitled to
//! know which one was taken:
//!
//! * **cube** — [`crate::scene::box_uv`] itself, called rather than transcribed.
//!   A unit cube is the one shape a dominant-axis box projection was already
//!   exactly right for, so this is the shape whose appearance must not move, and
//!   calling the function is a fact where a matching table would be a
//!   coincidence waiting to rot;
//! * **sphere** — equirectangular: `u` = longitude, `v` = latitude from the north
//!   pole. The seam is the `θ = 0` meridian, which is where the duplicated
//!   `slices + 1`th column already sat;
//! * **plane** — the XZ footprint itself, `u = x + ½`, `v = z + ½`;
//! * **cylinder** — the side unwraps to `u` = angle, `v` = height; each cap is a
//!   disc mapped into the unit square by its own `(cos, sin)`;
//! * **cone** — the same, with `v = 0` at the apex.
//!
//! All of them are **derived from the values the generator already computed**
//! (the `psin64`/`pcos64` ring, the face tangents, the corner signs), so the
//! no-`std`-trig law below reaches the uv for free.
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

    /// This kind's CPU geometry (`pos + normal + uv`, CCW-outside).
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

/// **The discriminant IS the dense index** — pinned, because two spellings of it
/// are in use and one of them crosses into WGSL (the I4b audit).
///
/// `pack_fallback` buckets with [`PrimMesh::index`]; `ScatterData::build` folds
/// `mesh as u32` into the content key; and island wave I4b's `ScatterNode` puts
/// `mesh as u32` in `RasterParamsGpu::material.z`, where `impostor_radius` in
/// `scatter_mesh.wgsl` branches on it to size an impostor card — `0u` cube, `1u`
/// sphere, `2u` plane, anything else the cylinder/cone hypotenuse. `PrimMesh`
/// carries no `#[repr]` and no explicit discriminants, so today the two agree by
/// declaration order alone; adding an explicit discriminant, or inserting a kind
/// anywhere but the end, would silently give every cube a sphere's billboard.
/// A compile-time assertion is the cheapest place to notice.
const _: () = {
    let mut i = 0;
    while i < PrimMesh::ALL.len() {
        assert!(PrimMesh::ALL[i] as u32 as usize == i);
        assert!(PrimMesh::ALL[i].index() == i);
        i += 1;
    }
};

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
fn v(pos: [f32; 3], normal: [f32; 3], uv: [f32; 2]) -> MeshVertex {
    MeshVertex { pos, normal, uv }
}

/// A disc's `[0,1]²` cap mapping from the ring's own `(cos, sin)` — the cylinder's
/// two caps and the cone's base, written once so the three agree.
///
/// `v` is flipped against `sin` so the cap reads the same way round as the side
/// unwrap does; a cap is a decision either way, and one decision is better than
/// three.
#[inline]
fn cap_uv(c: f32, s: f32) -> [f32; 2] {
    [0.5 + 0.5 * c, 0.5 - 0.5 * s]
}

/// Unit cube centred at the origin (extent `±0.5`), 24 verts / 36 indices. Moved
/// here from `passes::mesh` (re-exported there for existing references).
/// Positions, normals and winding are unchanged since R-P1 — P26.5 appended a
/// `uv`, which no textureless scene reads, so every cube golden stays
/// identical.
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
            // P26.5: the cube's parametrization **is** the projection it
            // replaces, computed by the same function rather than transcribed
            // from it. A unit cube is the one shape a dominant-axis box
            // projection was already exactly right for, so this is the shape
            // whose appearance must not move — and `crate::scene::box_uv`
            // agreeing with a hand-written table is a coincidence waiting to
            // rot, while calling it is a fact. (The ±Y faces are why: their
            // tangent pair runs `v` along −Z, so the corner-sign formula the
            // other four faces satisfy is inverted on exactly two of six.)
            verts.push(v(p.to_array(), n, crate::scene::box_uv(p.to_array(), n)));
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
            // P26.5, equirectangular: `u` is longitude and `v` latitude from the
            // north pole. The `slices + 1`th column is the duplicated seam
            // vertex, which is precisely why it exists — it takes `u = 1` while
            // its twin takes `u = 0`.
            verts.push(v(
                [n[0] * 0.5, n[1] * 0.5, n[2] * 0.5],
                n,
                [j as f32 / slices as f32, i as f32 / stacks as f32],
            ));
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
    // P26.5: the XZ footprint itself — `u = x + ½`, `v = z + ½`. A plane is the
    // one shape where any other choice would be arbitrary.
    let verts = vec![
        v([-0.5, 0.0, 0.5], n, [0.0, 1.0]),
        v([0.5, 0.0, 0.5], n, [1.0, 1.0]),
        v([0.5, 0.0, -0.5], n, [1.0, 0.0]),
        v([-0.5, 0.0, -0.5], n, [0.0, 0.0]),
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

    // Side rings: bottom `[0..=segs]`, top `[segs+1 ..= 2·segs+1]`. Radial
    // normals.
    //
    // **`segs + 1` columns, not `segs`** (P26.5). The ring used to be
    // wrap-SHARED — the last quad's right edge was column 0 again — which is
    // free while there is no uv and *wrong* the moment there is one: the seam
    // segment would run `u` from `(segs-1)/segs` back to `0`, i.e. the whole
    // texture mirrored into one twenty-fourth of the barrel. The extra column is
    // the same duplicated seam vertex `sphere_geometry` has always emitted, for
    // the same reason. Positions and normals are unchanged, so the surface is
    // the surface it was.
    let ring: Vec<(f32, f32)> = (0..segs)
        .map(|s| {
            let a = TAU64 * s as f64 / segs as f64;
            (pcos64(a) as f32, psin64(a) as f32)
        })
        .collect();
    let cols = segs + 1;
    for i in 0..cols {
        let (c, s) = ring[(i % segs) as usize];
        let u = i as f32 / segs as f32;
        verts.push(v([0.5 * c, -0.5, 0.5 * s], [c, 0.0, s], [u, 1.0]));
    }
    for i in 0..cols {
        let (c, s) = ring[(i % segs) as usize];
        let u = i as f32 / segs as f32;
        verts.push(v([0.5 * c, 0.5, 0.5 * s], [c, 0.0, s], [u, 0.0]));
    }
    for s in 0..segs {
        let (bc, bn) = (s as u16, (s + 1) as u16);
        let (tc, tn) = ((cols + s) as u16, (cols + s + 1) as u16);
        indices.extend_from_slice(&[bc, tc, bn, bn, tc, tn]);
    }

    // Top cap (+Y): centre + rim fan.
    let top_center = verts.len() as u16;
    verts.push(v([0.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5]));
    let top_rim = verts.len() as u16;
    for &(c, s) in &ring {
        verts.push(v([0.5 * c, 0.5, 0.5 * s], [0.0, 1.0, 0.0], cap_uv(c, s)));
    }
    for s in 0..segs {
        let s1 = (s + 1) % segs;
        indices.extend_from_slice(&[top_center, top_rim + s1 as u16, top_rim + s as u16]);
    }

    // Bottom cap (−Y): centre + rim fan (reversed winding for the −Y normal).
    let bot_center = verts.len() as u16;
    verts.push(v([0.0, -0.5, 0.0], [0.0, -1.0, 0.0], [0.5, 0.5]));
    let bot_rim = verts.len() as u16;
    for &(c, s) in &ring {
        verts.push(v([0.5 * c, -0.5, 0.5 * s], [0.0, -1.0, 0.0], cap_uv(c, s)));
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
        // P26.5: the side unwraps like the cylinder's — `u` = angle, `v` = 0 at
        // the apex. The cone needs no duplicated seam column because every
        // segment already has its own three vertices, so `s = segs - 1` ends at
        // `u = 1` rather than wrapping to 0.
        let (u0, u1) = (s as f32 / segs as f32, (s + 1) as f32 / segs as f32);
        verts.push(v([0.0, 0.5, 0.0], n_apex, [0.5 * (u0 + u1), 0.0])); // apex
        verts.push(v(r1, n1, [u1, 1.0])); // rim[s+1]
        verts.push(v(r0, n0, [u0, 1.0])); // rim[s]
        indices.extend_from_slice(&[base, base + 1, base + 2]);
    }

    // Base cap (−Y): centre + rim fan.
    let center = verts.len() as u16;
    verts.push(v([0.0, -0.5, 0.0], [0.0, -1.0, 0.0], [0.5, 0.5]));
    let rim = verts.len() as u16;
    for s in 0..segs {
        let a = angle(s);
        let (c, sn) = (pcos64(a) as f32, psin64(a) as f32);
        verts.push(v(
            [0.5 * c, -0.5, 0.5 * sn],
            [0.0, -1.0, 0.0],
            cap_uv(c, sn),
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

    /// Bind the shared vertex + index buffers and **nothing else** — for a pass
    /// that pulls its instance data from a storage buffer instead of a vertex-step
    /// one (P27.2's page raster, whose instance count comes out of an indirect
    /// args block the GPU wrote).
    pub fn bind_geometry<'p>(&'p self, pass: &mut wgpu::RenderPass<'p>) {
        pass.set_vertex_buffer(0, self.vertices.slice(..));
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint16);
    }

    /// Where one kind's indices live in the shared buffers — the three constant
    /// words of a `draw_indexed_indirect` args block.
    #[inline]
    pub fn range(&self, kind: PrimMesh) -> PrimRange {
        self.ranges[kind.index()]
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
/// with **six** floats per vertex (position then normal), and `indices` widens
/// the packed `u16` index list to `u32` because WGSL has no 16-bit scalar type.
/// Widening costs ~4 KB for the whole primitive set and buys an index read that
/// is one array subscript.
///
/// **Six and not eight** (P26.5): [`MeshVertex`] grew a `uv` for the virtual
/// texture path, and `scatter_mesh.wgsl` pulls with a hard-coded stride of six.
/// The scatter path samples no virtual texture — a scattered instance carries no
/// [`crate::VtTextureSet`] at all — so widening this buffer would cost every
/// foliage vertex two floats to feed a branch that cannot be taken.
///
/// **The P26.5 audit made that a pin instead of an argument.** The original
/// wording said the flatten "names the two fields it copies rather than casting
/// the struct, so the stride and the copy cannot drift apart" — which is a
/// description of the code, not a check on it. Measured: adding
/// `flat.extend_from_slice(&v.uv)` here draws every scattered instance from the
/// wrong bytes and the **whole `inf-render` suite stays green**, all four
/// scatter goldens included, because the goldens' pixel comparison is opt-in.
/// Before P26.5 the drift was impossible (a `MeshVertex` *was* six floats); the
/// uv is what made it a hazard, so [`SCATTER_PULL_STRIDE`] and
/// `the_scatter_pull_buffer_is_the_stride_the_shader_pulls_with` exist.
///
/// It exists rather than reusing [`PrimGpu`]'s buffers because a wgpu buffer's
/// usage flags are fixed at creation and `VERTEX | INDEX` is not `STORAGE`; the
/// geometry itself is byte-identical, produced by the same [`packed_geometry`].
pub struct PrimStorage {
    pub vertices: wgpu::Buffer,
    pub indices: wgpu::Buffer,
    pub ranges: [PrimRange; 5],
}

/// Floats per vertex in [`PrimStorage`]'s pull buffer — **six**: position then
/// normal, and deliberately not [`MeshVertex`]'s uv (P26.5).
///
/// `scatter_mesh.wgsl` indexes that buffer as `vertices[idx * 6u + k]`. WGSL has
/// no way to import a Rust constant, so the agreement is pinned from this side
/// instead: the arm below asserts the flatten produces exactly this many floats
/// per vertex **and** that the shader spells the same number.
pub const SCATTER_PULL_STRIDE: usize = 6;

/// The pull buffer's contents: `position` then `normal` per vertex, flat.
///
/// A named function rather than a loop inside `PrimStorage::new` so a test can
/// run it with no GPU — the whole point of the P26.5 audit's finding is that the
/// only consumer of these bytes is a shader, on a path whose pixels nothing
/// compares by default.
fn scatter_pull_floats(verts: &[MeshVertex]) -> Vec<f32> {
    let mut flat: Vec<f32> = Vec::with_capacity(verts.len() * SCATTER_PULL_STRIDE);
    for v in verts {
        flat.extend_from_slice(&v.pos);
        flat.extend_from_slice(&v.normal);
    }
    flat
}

impl PrimStorage {
    pub fn new(gpu: &GpuContext, label: &str) -> Self {
        let (verts, idx, ranges) = packed_geometry();
        let flat = scatter_pull_floats(&verts);
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

    // ── the uv streams (P26.5) ───────────────────────────────────────────────

    /// **Every built-in shape carries a real parametrization** — inside the unit
    /// square, and *spanning* it on both axes.
    ///
    /// The span is the load-bearing half and the P23 law says why: *"every UV is
    /// inside the unit square" is satisfied perfectly by all-zeros.* A generator
    /// that forgot its uv, or a `v()` helper defaulting one, passes a bounds
    /// check and fails this. The floor is 0.9 rather than 1.0 because the
    /// sphere's poles and the caps' discs do not reach a corner.
    #[test]
    fn every_primitive_spans_its_own_uv_square() {
        for kind in PrimMesh::ALL {
            let (verts, _) = kind.geometry();
            let mut lo = [f32::MAX; 2];
            let mut hi = [f32::MIN; 2];
            for v in &verts {
                for a in 0..2 {
                    assert!(
                        (-1e-6..=1.000_001).contains(&v.uv[a]),
                        "{kind:?}: uv {:?} leaves the unit square",
                        v.uv
                    );
                    lo[a] = lo[a].min(v.uv[a]);
                    hi[a] = hi[a].max(v.uv[a]);
                }
            }
            for a in 0..2 {
                assert!(
                    hi[a] - lo[a] > 0.9,
                    "{kind:?}: uv axis {a} spans only {} — an unfilled stream \
                     (all zeros) looks exactly like this and passes every bounds \
                     check there is",
                    hi[a] - lo[a]
                );
            }
        }
    }

    /// **The cube's uv is the retired box projection, face for face.**
    ///
    /// The one shape whose appearance must not move: `vt_box_uv` mapped a unit
    /// cube's faces onto `[0,1]²` and so does the stream, so a surface that
    /// sampled this way through P26.3/P26.4 samples the same texels now. Any
    /// other shape *does* move, deliberately — that is what "a box projection on
    /// a character is visibly wrong" means.
    #[test]
    fn the_cube_uv_is_the_projection_it_replaces() {
        let (verts, _) = cube_geometry();
        for v in &verts {
            let want = crate::scene::box_uv(v.pos, v.normal);
            assert!(
                (v.uv[0] - want[0]).abs() < 1e-6 && (v.uv[1] - want[1]).abs() < 1e-6,
                "cube vertex {:?} (normal {:?}) is uv {:?}, the box projection says {want:?}",
                v.pos,
                v.normal,
                v.uv
            );
        }
    }

    /// **The cylinder's side has a seam column, not a shared one.**
    ///
    /// The ring was wrap-shared before P26.5 — free without a uv, and with one it
    /// means the last quad runs `u` from `(segs-1)/segs` back to `0`: the whole
    /// texture mirrored into one twenty-fourth of the barrel. Asserted as "every
    /// side quad's `u` increases", which is what the shared column cannot do,
    /// plus the two anti-vacuity facts that make the sweep mean something (the
    /// quads exist, and one of them really is the wrap).
    #[test]
    fn the_cylinder_side_never_runs_its_uv_backwards() {
        let (verts, indices) = cylinder_geometry();
        let segs = RADIAL_SEGMENTS as usize;
        let cols = segs + 1;
        // The side occupies the first `2·cols` vertices and the first `6·segs`
        // indices, by construction above.
        let mut checked = 0usize;
        for t in indices[..segs * 6].chunks(3) {
            let mut us: Vec<f32> = t.iter().map(|i| verts[*i as usize].uv[0]).collect();
            us.sort_by(f32::total_cmp);
            assert!(
                us[2] - us[0] <= 1.0 / segs as f32 + 1e-6,
                "a side triangle spans {} of u — the seam column is shared, so the \
                 texture is mirrored across one segment",
                us[2] - us[0]
            );
            checked += 1;
        }
        assert_eq!(checked, segs * 2, "the side quads were not swept");
        // The wrap really is there: the last side column is u = 1 and the first
        // is u = 0, over two vertices at the SAME position.
        assert_eq!(verts[0].uv[0], 0.0);
        assert_eq!(verts[cols - 1].uv[0], 1.0);
        assert_eq!(verts[0].pos, verts[cols - 1].pos);
    }

    /// **The scatter pull buffer's stride is the stride the shader pulls with**
    /// (P26.5 audit).
    ///
    /// `scatter_mesh.wgsl` reads this buffer as `vertices[idx * 6u + k]`, and
    /// WGSL cannot import a Rust constant. Until P26.5 the agreement was
    /// structural — a `MeshVertex` was six floats, so a `cast_slice` and a
    /// field-by-field copy produced the same bytes. The uv broke that: the
    /// struct is eight floats now and the buffer must stay six.
    ///
    /// Measured, which is why this arm exists: adding
    /// `flat.extend_from_slice(&v.uv)` to the flatten draws every scattered
    /// instance from the wrong bytes, and the **whole `inf-render` suite is
    /// green** — including all four scatter goldens, because a golden's pixel
    /// comparison is opt-in (`INF_GOLDEN_STRICT`).
    ///
    /// Both directions: the flatten's length and contents, and the literal in
    /// the shader. Changing one without the other fails here.
    #[test]
    fn the_scatter_pull_buffer_is_the_stride_the_shader_pulls_with() {
        let (verts, _, _) = packed_geometry();
        let flat = scatter_pull_floats(&verts);
        assert_eq!(
            flat.len(),
            verts.len() * SCATTER_PULL_STRIDE,
            "the pull buffer is {} floats for {} vertices — the shader indexes it \
             at {SCATTER_PULL_STRIDE} per vertex and would read every one of them \
             from another vertex's bytes",
            flat.len(),
            verts.len()
        );
        // …and the six really are position then normal, in that order. A stride
        // that matched with the fields permuted is the same defect one step
        // quieter.
        for (i, v) in verts.iter().enumerate().take(8) {
            let base = i * SCATTER_PULL_STRIDE;
            assert_eq!(&flat[base..base + 3], &v.pos[..], "vertex {i} position");
            assert_eq!(
                &flat[base + 3..base + 6],
                &v.normal[..],
                "vertex {i} normal"
            );
        }
        // The other side of the agreement: the shader's own literal.
        let src = include_str!("shaders/scatter_mesh.wgsl");
        let spelled = format!("idx * {SCATTER_PULL_STRIDE}u");
        assert!(
            src.contains(&spelled),
            "`scatter_mesh.wgsl` does not pull at `{spelled}`, so the buffer this \
             crate writes and the buffer that shader reads are two different \
             layouts"
        );
        // ANTI-VACUITY: a `MeshVertex` is WIDER than the stride, so this is a
        // statement about a deliberate omission and not about a coincidence.
        assert!(
            std::mem::size_of::<MeshVertex>() / std::mem::size_of::<f32>() > SCATTER_PULL_STRIDE,
            "a MeshVertex is no wider than the pull stride, so nothing is being \
             left out and this arm is checking a tautology"
        );
    }

    /// The box projection is a **table**, not a description: three faces, three
    /// dominant axes, and the `[0,1]²` corner each lands on.
    ///
    /// It is the last fallback for a surface with no authored uv
    /// (`deformed_skinned_mesh`), so it is pinned by value rather than by the
    /// shader it was transliterated from — that shader is gone.
    #[test]
    fn the_box_projection_is_the_one_it_always_was() {
        use crate::scene::box_uv;
        // +Z face, the four corners of a unit quad at z = +0.5.
        assert_eq!(box_uv([-0.5, -0.5, 0.5], [0.0, 0.0, 1.0]), [0.0, 1.0]);
        assert_eq!(box_uv([0.5, 0.5, 0.5], [0.0, 0.0, 1.0]), [1.0, 0.0]);
        // +X face: `u` runs along −Z.
        assert_eq!(box_uv([0.5, -0.5, -0.5], [1.0, 0.0, 0.0]), [1.0, 1.0]);
        // +Y face: `u` runs along X, `v` along −Z.
        assert_eq!(box_uv([-0.5, 0.5, 0.5], [0.0, 1.0, 0.0]), [0.0, 0.0]);
        // A degenerate normal picks the X branch and still returns a number.
        assert!(box_uv([0.1, 0.2, 0.3], [0.0; 3])
            .iter()
            .all(|f| f.is_finite()));
    }
}
