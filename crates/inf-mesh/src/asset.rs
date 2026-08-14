//! The `.inf_mesh` payload schema.
//!
//! A mesh asset is one or more submeshes of interleaved vertices + a 32-bit
//! index buffer, plus a local-space bounding box. Vertices are interleaved
//! (position/normal/uv/tangent) because that is the layout the renderer uploads
//! directly and the layout `meshopt`'s vertex-fetch optimization assumes.

use bytemuck::{Pod, Zeroable};
use inf_asset::{AssetKind, AssetPayload};
use serde::{Deserialize, Serialize};

/// **The tangent an importer writes when the source file has none** — `[1, 0, 0, 1]`.
///
/// Named once, here, because it is not a direction: it is the absence of one,
/// spelled as a value so [`MeshVertex`] can stay `Pod`. Every producer in the
/// tree writes exactly this (`gltf_import`'s `unwrap_or`, `obj_import`
/// unconditionally, `inf_dcc::TANGENT_FALLBACK` for a corner with no usable
/// accumulation, and [`MeshVertex::default`]), and until P28.2 nothing on the
/// meshlet path read the field, so the placeholder cost nothing.
///
/// It costs something now, which is why it has a name (P28.2 audit): a v3 cook
/// packs whatever is in this field into `VgeomVertex::tangent`, and
/// `pack_tangent` returns its `NO_TANGENT` sentinel only for a *non-finite or
/// zero-length* input. `[1, 0, 0]` is neither. So an OBJ mesh, a glTF with no
/// `TANGENT` attribute, or a UV-less DCC export would shade through a constant
/// object-space +X tangent instead of the per-fragment derivative frame it used
/// before the channel existed — a visible change to a surface whose author
/// supplied nothing. [`MeshAsset::vgeom_streams`] is where that is stopped.
pub const TANGENT_PLACEHOLDER: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

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
            tangent: TANGENT_PLACEHOLDER,
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
    ///
    /// # `sum > 1e-6` catches NaN and lets `+inf` through (C4-37)
    ///
    /// The guard reads as though it rejects everything unusual, and it half
    /// does: `NaN > 1e-6` is false, so a NaN weight takes the safe fallback.
    /// `inf > 1e-6` is **true**, so `weights = [inf, 0, 0, 0]` passes and each
    /// division is `inf / inf` — NaN, manufactured out of an input that never
    /// held one. It is then serialized into the `.inf_mesh` skin stream, which
    /// is `#[repr(C)] Pod` and uploaded straight to a GPU vertex buffer.
    ///
    /// The importers refuse a non-finite `WEIGHTS_0` at the door
    /// ([`crate::validate`]), so this is the second line rather than the first;
    /// it stays because `VertexSkin` is public and a caller who built one by
    /// hand never crossed that door.
    pub fn normalized(mut self) -> Self {
        let sum: f32 = self.weights.iter().sum();
        if sum.is_finite() && sum > 1e-6 && self.weights.iter().all(|w| w.is_finite()) {
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
    /// tangents, indices)` geometry — the raw streams the virtualized-geometry
    /// builder (`inf-vgeom`) consumes at cook time. Submesh index buffers are
    /// rebased onto the concatenated vertex buffer.
    ///
    /// v1 fuses all material slots into one geometry (virtualized geometry treats
    /// the whole mesh as one clusterizable surface); per-slot meshlet tagging is a
    /// documented follow-up. Additive: does not change any payload.
    ///
    /// **The tangent stream joined in P28.2**, where `.inf_vmesh` v3 gave
    /// `VgeomVertex` somewhere to put one. It has been sitting in
    /// [`MeshVertex::tangent`] since P4 — read by nothing that reaches the meshlet
    /// path, which is exactly the gap `docs/memos/p26-5-vertex-streams.md`
    /// recorded and routed here.
    ///
    /// **A mesh whose every vertex carries [`TANGENT_PLACEHOLDER`] hands back an
    /// EMPTY tangent stream** (P28.2 audit), which `build_vgeom` writes as
    /// `NO_TANGENT` and the shader reads as "use the derivative frame". The
    /// alternative was measured and is a regression: the importers substitute
    /// `[1, 0, 0, 1]` when a source file has no tangents, `pack_tangent` refuses
    /// only a non-finite or zero-length input, so every OBJ mesh and every
    /// untangented glTF would have started shading through a constant +X tangent
    /// the moment the container went to v3. The test is over the WHOLE mesh and
    /// exact — never per vertex and never a tolerance — and it cannot misfire: a
    /// surface whose authored tangent really is uniformly `+X` has axis-aligned
    /// uvs, and the derivative frame derives `+X` for it too.
    #[allow(clippy::type_complexity)]
    pub fn vgeom_streams(
        &self,
    ) -> (
        Vec<[f32; 3]>,
        Vec<[f32; 3]>,
        Vec<[f32; 2]>,
        Vec<[f32; 4]>,
        Vec<u32>,
    ) {
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut tangents = Vec::new();
        let mut indices = Vec::new();
        for sm in &self.submeshes {
            let base = positions.len() as u32;
            for v in &sm.vertices {
                positions.push(v.position);
                normals.push(v.normal);
                uvs.push(v.uv);
                tangents.push(v.tangent);
            }
            indices.extend(sm.indices.iter().map(|&i| i + base));
        }
        if tangents.iter().all(|t| *t == TANGENT_PLACEHOLDER) {
            tangents.clear();
        }
        (positions, normals, uvs, tangents, indices)
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

    /// **The placeholder is not a direction, and `vgeom_streams` must not ship it
    /// as one** (P28.2 audit). Exact and whole-mesh: one authored tangent
    /// anywhere keeps the whole stream, because the placeholders beside it are
    /// then corners of a real field rather than the absence of one.
    #[test]
    fn an_untangented_mesh_hands_the_vgeom_builder_no_tangent_stream() {
        let quad = |tangent: Option<[f32; 4]>| {
            let mut v = vec![MeshVertex::default(); 4];
            if let Some(t) = tangent {
                v[2].tangent = t;
            }
            MeshAsset::new(
                vec![SubMesh {
                    name: "q".into(),
                    vertices: v,
                    indices: vec![0, 1, 2, 0, 2, 3],
                    material_slot: None,
                    skin: Vec::new(),
                }],
                Vec::new(),
            )
        };

        let (_, _, _, tangents, _) = quad(None).vgeom_streams();
        assert!(
            tangents.is_empty(),
            "a mesh of nothing but placeholders offered the builder a direction"
        );

        let authored = [0.0, 0.0, 1.0, -1.0];
        let (_, _, _, tangents, _) = quad(Some(authored)).vgeom_streams();
        assert_eq!(
            tangents.len(),
            4,
            "one authored tangent must keep the whole stream"
        );
        assert_eq!(tangents[2], authored);
        assert_eq!(tangents[0], TANGENT_PLACEHOLDER);
    }

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
