//! The `.inf_mesh` payload schema.
//!
//! A mesh asset is one or more submeshes of interleaved vertices + a 32-bit
//! index buffer, plus a local-space bounding box. Vertices are interleaved
//! (position/normal/uv/tangent) because that is the layout the renderer uploads
//! directly and the layout `meshopt`'s vertex-fetch optimization assumes.

use bytemuck::{Pod, Zeroable};
use inf_asset::{AssetKind, AssetPayload};
use serde::{Deserialize, Serialize};

/// One interleaved vertex. `#[repr(C)]` + `Pod` so it uploads to a GPU buffer
/// and feeds `meshopt` without a copy. 48 bytes, naturally aligned (no padding).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    /// xyz = tangent, w = handedness sign (±1) for the bitangent.
    pub tangent: [f32; 4],
}

impl Default for MeshVertex {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            normal: [0.0, 1.0, 0.0],
            uv: [0.0; 2],
            tangent: [1.0, 0.0, 0.0, 1.0],
        }
    }
}

/// Per-vertex skinning influences (P11.1): the four joints that deform a vertex
/// and their **normalized** weights (`Σ = 1`). Stored as a **parallel stream**
/// to [`SubMesh::vertices`] (index-aligned) rather than folded into
/// [`MeshVertex`] so the existing 48-byte interleaved vertex — and every
/// `.inf_mesh` payload written before skinning — is untouched. `#[repr(C)]` +
/// `Pod` so the renderer uploads it straight to a GPU vertex buffer. 24 bytes,
/// naturally aligned (no padding).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct VertexSkin {
    /// Joint indices into the mesh's skeleton (glTF `JOINTS_0`, widened to u16).
    pub joints: [u16; 4],
    /// Influence weights (glTF `WEIGHTS_0`), normalized so they sum to 1.
    pub weights: [f32; 4],
}

impl Default for VertexSkin {
    fn default() -> Self {
        Self {
            joints: [0; 4],
            weights: [1.0, 0.0, 0.0, 0.0],
        }
    }
}

impl VertexSkin {
    /// Normalize the weights so they sum to 1 (falls back to "all joint 0" when
    /// the weights are degenerate / zero).
    pub fn normalized(mut self) -> Self {
        let sum: f32 = self.weights.iter().sum();
        if sum > 1e-6 {
            for w in &mut self.weights {
                *w /= sum;
            }
        } else {
            self.weights = [1.0, 0.0, 0.0, 0.0];
        }
        self
    }
}

/// An axis-aligned bounding box in the mesh's local space (render f32).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Aabb {
    /// An empty box that grows to fit inserted points.
    pub fn empty() -> Self {
        Self {
            min: [f32::INFINITY; 3],
            max: [f32::NEG_INFINITY; 3],
        }
    }

    pub fn grow(&mut self, p: [f32; 3]) {
        for ((mn, mx), &pv) in self.min.iter_mut().zip(self.max.iter_mut()).zip(p.iter()) {
            *mn = mn.min(pv);
            *mx = mx.max(pv);
        }
    }

    pub fn from_points(points: impl IntoIterator<Item = [f32; 3]>) -> Self {
        let mut b = Self::empty();
        for p in points {
            b.grow(p);
        }
        if b.min[0] > b.max[0] {
            // No points: collapse to origin.
            b = Aabb {
                min: [0.0; 3],
                max: [0.0; 3],
            };
        }
        b
    }

    pub fn center(&self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    /// Radius of the bounding sphere around the box center — used to frame the
    /// mesh for thumbnails and F-focus.
    pub fn radius(&self) -> f32 {
        let c = self.center();
        let dx = self.max[0] - c[0];
        let dy = self.max[1] - c[1];
        let dz = self.max[2] - c[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// One drawable submesh: an interleaved vertex buffer + indices, tagged with the
/// material slot it should draw with (an index into the mesh's material slot
/// list, resolved to a real material asset by the importer).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubMesh {
    pub name: String,
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    /// Material slot index (glTF primitive material), or `None` for default.
    pub material_slot: Option<u32>,
    /// Per-vertex skinning influences (P11.1), index-aligned to `vertices`.
    /// **Additive field:** `#[serde(default)]` → an empty vec for every pre-P11
    /// payload and every static (unskinned) primitive, so this submesh is a rigid
    /// mesh unless a skin stream is present. When non-empty, `skin.len() ==
    /// vertices.len()`.
    #[serde(default)]
    pub skin: Vec<VertexSkin>,
}

impl SubMesh {
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }
    /// Whether this submesh carries per-vertex skinning influences.
    pub fn is_skinned(&self) -> bool {
        !self.skin.is_empty()
    }
}

/// The `.inf_mesh` payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshAsset {
    pub schema_version: u32,
    pub submeshes: Vec<SubMesh>,
    pub bounds: Aabb,
    /// Names of the material slots this mesh expects, in slot order. The
    /// importer maps these to material asset GUIDs (stored as dependencies).
    pub material_slots: Vec<String>,
}

impl MeshAsset {
    /// Schema v2 (P11.1): [`SubMesh`] gained the additive, `#[serde(default)]`
    /// `skin` stream. A v1 payload has no skinned submeshes; new payloads always
    /// write the (possibly empty) stream. No custom migration is needed beyond
    /// the default newer-than-current guard.
    pub const CURRENT_VERSION: u32 = 2;

    /// Assemble a mesh from submeshes, computing the overall bounds.
    pub fn new(submeshes: Vec<SubMesh>, material_slots: Vec<String>) -> Self {
        let bounds = Aabb::from_points(
            submeshes
                .iter()
                .flat_map(|s| s.vertices.iter().map(|v| v.position)),
        );
        Self {
            schema_version: Self::CURRENT_VERSION,
            submeshes,
            bounds,
            material_slots,
        }
    }

    pub fn triangle_count(&self) -> usize {
        self.submeshes.iter().map(SubMesh::triangle_count).sum()
    }
    pub fn vertex_count(&self) -> usize {
        self.submeshes.iter().map(SubMesh::vertex_count).sum()
    }

    /// Flatten every submesh into one combined `(positions, normals, uvs,
    /// indices)` geometry — the raw streams the virtualized-geometry builder
    /// (`inf-vgeom`) consumes at cook time. Submesh index buffers are rebased
    /// onto the concatenated vertex buffer.
    ///
    /// v1 fuses all material slots into one geometry (virtualized geometry treats
    /// the whole mesh as one clusterizable surface); per-slot meshlet tagging is a
    /// documented follow-up. Additive: does not change any payload.
    #[allow(clippy::type_complexity)]
    pub fn vgeom_streams(&self) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 2]>, Vec<u32>) {
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut indices = Vec::new();
        for sm in &self.submeshes {
            let base = positions.len() as u32;
            for v in &sm.vertices {
                positions.push(v.position);
                normals.push(v.normal);
                uvs.push(v.uv);
            }
            indices.extend(sm.indices.iter().map(|&i| i + base));
        }
        (positions, normals, uvs, indices)
    }
}

impl AssetPayload for MeshAsset {
    const KIND: AssetKind = AssetKind::Mesh;
    const SCHEMA_VERSION: u32 = Self::CURRENT_VERSION;
    fn schema_version(&self) -> u32 {
        self.schema_version
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_asset::{decode, encode};

    fn quad() -> SubMesh {
        let v = |x: f32, y: f32| MeshVertex {
            position: [x, y, 0.0],
            ..Default::default()
        };
        SubMesh {
            name: "quad".into(),
            vertices: vec![v(0.0, 0.0), v(1.0, 0.0), v(1.0, 1.0), v(0.0, 1.0)],
            indices: vec![0, 1, 2, 0, 2, 3],
            material_slot: Some(0),
            skin: Vec::new(),
        }
    }

    #[test]
    fn bounds_and_counts() {
        let m = MeshAsset::new(vec![quad()], vec!["Default".into()]);
        assert_eq!(m.triangle_count(), 2);
        assert_eq!(m.vertex_count(), 4);
        assert_eq!(m.bounds.min, [0.0, 0.0, 0.0]);
        assert_eq!(m.bounds.max, [1.0, 1.0, 0.0]);
    }

    #[test]
    fn payload_round_trips_deterministically() {
        let m = MeshAsset::new(vec![quad()], vec!["Default".into()]);
        let a = encode(&m).unwrap();
        let b = encode(&m).unwrap();
        assert_eq!(a, b);
        let back: MeshAsset = decode(&a).unwrap();
        assert_eq!(back, m);
    }
}
