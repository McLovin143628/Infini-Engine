//! glTF/GLB import → geometry + material/texture descriptors.
//!
//! This reads a glTF document and produces a Ring-0-friendly intermediate:
//!   * one [`MeshAsset`] per glTF mesh (primitives → submeshes, optimized);
//!   * [`ImportedMaterial`] descriptors (PBR factors + which image each map is);
//!   * [`RawImage`]s decoded to RGBA8 (glTF decodes via the `image` crate).
//!
//! The editor's import orchestrator turns this into real `.inf_mesh` /
//! `.inf_tex` / `.inf_mat` assets with GUIDs and dependency edges. Missing
//! normals are generated (flat per-face); missing tangents default (the material
//! pass can regenerate from UVs later).

use std::path::Path;

use glam::Vec3;

use crate::asset::{MeshAsset, MeshVertex, SubMesh};
use crate::error::MeshError;
use crate::optimize::optimize;

/// A decoded texture source, RGBA8, straight from the glTF image list.
#[derive(Debug, Clone, PartialEq)]
pub struct RawImage {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

/// A glTF material's PBR parameters + the image indices of its maps.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedMaterial {
    pub name: String,
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive: [f32; 3],
    /// Indices into [`GltfImport::images`], if the map is present.
    pub base_color_image: Option<usize>,
    pub normal_image: Option<usize>,
    pub metallic_roughness_image: Option<usize>,
}

/// One imported mesh with its display name.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedMesh {
    pub name: String,
    pub mesh: MeshAsset,
}

/// The full result of importing a glTF file.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GltfImport {
    pub meshes: Vec<ImportedMesh>,
    pub materials: Vec<ImportedMaterial>,
    pub images: Vec<RawImage>,
}

/// Import a glTF/GLB file.
pub fn import_gltf(path: &Path) -> Result<GltfImport, MeshError> {
    let (doc, buffers, images) =
        gltf::import(path).map_err(|e| MeshError::Gltf(format!("{path:?}: {e}")))?;

    let material_names: Vec<String> = doc
        .materials()
        .map(|m| m.name().unwrap_or("Material").to_string())
        .collect();

    let mut out = GltfImport::default();

    // Images → RGBA8.
    for (i, img) in images.iter().enumerate() {
        out.images.push(RawImage {
            name: format!("image_{i}"),
            width: img.width,
            height: img.height,
            rgba8: to_rgba8(img),
        });
    }

    // Materials.
    for mat in doc.materials() {
        let pbr = mat.pbr_metallic_roughness();
        out.materials.push(ImportedMaterial {
            name: mat.name().unwrap_or("Material").to_string(),
            base_color: pbr.base_color_factor(),
            metallic: pbr.metallic_factor(),
            roughness: pbr.roughness_factor(),
            emissive: mat.emissive_factor(),
            base_color_image: pbr
                .base_color_texture()
                .map(|t| t.texture().source().index()),
            normal_image: mat.normal_texture().map(|t| t.texture().source().index()),
            metallic_roughness_image: pbr
                .metallic_roughness_texture()
                .map(|t| t.texture().source().index()),
        });
    }

    // Meshes → submeshes.
    for mesh in doc.meshes() {
        let name = mesh.name().unwrap_or("Mesh").to_string();
        let mut submeshes = Vec::new();
        for (pi, prim) in mesh.primitives().enumerate() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                // Only triangle lists are supported; skip lines/points.
                continue;
            }
            let reader = prim.reader(|b| Some(&buffers[b.index()]));
            let positions: Vec<[f32; 3]> = match reader.read_positions() {
                Some(p) => p.collect(),
                None => continue, // a primitive with no positions is unusable
            };
            let indices: Vec<u32> = match reader.read_indices() {
                Some(idx) => idx.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };
            let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|n| n.collect());
            let uvs: Option<Vec<[f32; 2]>> = reader
                .read_tex_coords(0)
                .map(|t| t.into_f32().collect::<Vec<_>>());
            let tangents: Option<Vec<[f32; 4]>> = reader.read_tangents().map(|t| t.collect());

            let normals = normals.unwrap_or_else(|| compute_normals(&positions, &indices));

            let mut verts = Vec::with_capacity(positions.len());
            for i in 0..positions.len() {
                verts.push(MeshVertex {
                    position: positions[i],
                    normal: *normals.get(i).unwrap_or(&[0.0, 1.0, 0.0]),
                    uv: uvs.as_ref().map(|u| u[i]).unwrap_or([0.0, 0.0]),
                    tangent: tangents
                        .as_ref()
                        .map(|t| t[i])
                        .unwrap_or([1.0, 0.0, 0.0, 1.0]),
                });
            }

            let (verts, idx) = optimize(verts, indices);
            submeshes.push(SubMesh {
                name: format!("{name}_{pi}"),
                vertices: verts,
                indices: idx,
                material_slot: prim.material().index().map(|i| i as u32),
            });
        }
        if submeshes.is_empty() {
            continue;
        }
        out.meshes.push(ImportedMesh {
            name: name.clone(),
            mesh: MeshAsset::new(submeshes, material_names.clone()),
        });
    }

    if out.meshes.is_empty() {
        return Err(MeshError::Gltf(format!("{path:?}: no triangle meshes")));
    }
    Ok(out)
}

/// Flat per-face normals for geometry that ships without them.
fn compute_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut acc = vec![Vec3::ZERO; positions.len()];
    for tri in indices.chunks_exact(3) {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let pa = Vec3::from(positions[a]);
        let pb = Vec3::from(positions[b]);
        let pc = Vec3::from(positions[c]);
        let n = (pb - pa).cross(pc - pa);
        acc[a] += n;
        acc[b] += n;
        acc[c] += n;
    }
    acc.into_iter()
        .map(|v| v.normalize_or_zero().to_array())
        .collect()
}

/// Expand a glTF image (any channel layout / bit depth we recognize) to RGBA8.
fn to_rgba8(img: &gltf::image::Data) -> Vec<u8> {
    use gltf::image::Format;
    let n = (img.width * img.height) as usize;
    let px = &img.pixels;
    let mut out = Vec::with_capacity(n * 4);
    match img.format {
        Format::R8 => {
            for &r in px.iter().take(n) {
                out.extend_from_slice(&[r, r, r, 255]);
            }
        }
        Format::R8G8 => {
            for c in px.chunks_exact(2).take(n) {
                out.extend_from_slice(&[c[0], c[0], c[0], c[1]]);
            }
        }
        Format::R8G8B8 => {
            for c in px.chunks_exact(3).take(n) {
                out.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
        }
        Format::R8G8B8A8 => {
            out.extend_from_slice(&px[..(n * 4).min(px.len())]);
        }
        Format::R16 | Format::R16G16 | Format::R16G16B16 | Format::R16G16B16A16 => {
            // 16-bit: take the high byte of each channel, expand to RGBA8.
            let channels = match img.format {
                Format::R16 => 1,
                Format::R16G16 => 2,
                Format::R16G16B16 => 3,
                _ => 4,
            };
            for texel in px.chunks_exact(channels * 2).take(n) {
                let mut rgba = [0u8, 0, 0, 255];
                for ch in 0..channels {
                    rgba[ch.min(3)] = texel[ch * 2 + 1]; // high byte (LE)
                }
                if channels == 1 {
                    rgba = [rgba[0], rgba[0], rgba[0], 255];
                }
                out.extend_from_slice(&rgba);
            }
        }
        Format::R32G32B32FLOAT | Format::R32G32B32A32FLOAT => {
            let channels = if matches!(img.format, Format::R32G32B32FLOAT) {
                3
            } else {
                4
            };
            for texel in px.chunks_exact(channels * 4).take(n) {
                let mut rgba = [0u8, 0, 0, 255];
                for ch in 0..channels {
                    let bytes = [
                        texel[ch * 4],
                        texel[ch * 4 + 1],
                        texel[ch * 4 + 2],
                        texel[ch * 4 + 3],
                    ];
                    let f = f32::from_le_bytes(bytes).clamp(0.0, 1.0);
                    rgba[ch.min(3)] = (f * 255.0) as u8;
                }
                out.extend_from_slice(&rgba);
            }
        }
    }
    // Pad if a truncated source left us short.
    out.resize(n * 4, 0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a minimal single-triangle glTF (external .bin buffer) and import it.
    #[test]
    fn imports_a_minimal_triangle_gltf() {
        let dir = tempfile::tempdir().unwrap();
        // Buffer: 3 positions (VEC3 f32) then 3 indices (u16).
        let positions: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let mut buf = Vec::new();
        for p in positions {
            for f in p {
                buf.extend_from_slice(&f.to_le_bytes());
            }
        }
        let idx_offset = buf.len();
        for i in [0u16, 1, 2] {
            buf.extend_from_slice(&i.to_le_bytes());
        }
        std::fs::write(dir.path().join("tri.bin"), &buf).unwrap();

        let gltf = format!(
            r#"{{
  "asset": {{ "version": "2.0" }},
  "scene": 0,
  "scenes": [{{ "nodes": [0] }}],
  "nodes": [{{ "mesh": 0 }}],
  "meshes": [{{ "name": "Tri", "primitives": [{{ "attributes": {{ "POSITION": 0 }}, "indices": 1 }}] }}],
  "buffers": [{{ "uri": "tri.bin", "byteLength": {total} }}],
  "bufferViews": [
    {{ "buffer": 0, "byteOffset": 0, "byteLength": {pos_len}, "target": 34962 }},
    {{ "buffer": 0, "byteOffset": {idx_offset}, "byteLength": 6, "target": 34963 }}
  ],
  "accessors": [
    {{ "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,1,0] }},
    {{ "bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR" }}
  ]
}}"#,
            total = buf.len(),
            pos_len = idx_offset,
            idx_offset = idx_offset,
        );
        let gltf_path = dir.path().join("tri.gltf");
        std::fs::write(&gltf_path, gltf).unwrap();

        let imported = import_gltf(&gltf_path).unwrap();
        assert_eq!(imported.meshes.len(), 1);
        let m = &imported.meshes[0].mesh;
        assert_eq!(m.triangle_count(), 1);
        assert_eq!(m.vertex_count(), 3);
        // Normals were generated (the source had none): +Z face.
        let n = m.submeshes[0].vertices[0].normal;
        assert!(n[2] > 0.9, "generated normal faces +Z, got {n:?}");
    }
}
