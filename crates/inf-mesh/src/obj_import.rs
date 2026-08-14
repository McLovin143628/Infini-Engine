//! Wavefront `.obj` import → the **same** [`GltfImport`] container the glTF path
//! produces, so the editor's import orchestrator needs zero new code paths.
//!
//! This is a hand-rolled parser (no `tobj` — the format subset we support is
//! small and a third-party dep would cost a license review). It covers the
//! geometry core plus a deliberately tiny slice of the companion `.mtl` format.
//!
//! # Container semantics (matches glTF exactly)
//!
//! glTF maps *mesh → [`ImportedMesh`]* and *primitive → [`SubMesh`]* (each
//! primitive carrying one material). We mirror that granularity:
//!
//! * an **`o` (object)** boundary starts a new [`ImportedMesh`] — the glTF-mesh
//!   analogue;
//! * a **`g` (group)** or **`usemtl`** boundary starts a new [`SubMesh`] within
//!   the current mesh — the glTF-primitive analogue.
//!
//! An `.obj` with no `o`/`g` at all imports as a single mesh named `"Mesh"`.
//! Every mesh shares the full ordered material-name list (`material_slots`) and
//! each submesh's `material_slot` indexes into it, exactly as the glTF importer
//! wires `material_names`.
//!
//! # UV convention
//!
//! The glTF importer reads `TEXCOORD_0` straight through (glTF's origin is the
//! **upper-left** corner). `.obj` `vt` uses a **lower-left** origin, so we flip
//! `v` (`v_out = 1 − v`) to land in the same space the rest of the engine
//! assumes.
//!
//! # Honest scope
//!
//! Geometry: `v` (extra `w`/vertex-color components ignored), `vn`, `vt`
//! (3rd `w` ignored); `f` in all reference forms (`v`, `v/vt`, `v//vn`,
//! `v/vt/vn`); convex polygon fan-triangulation (n ≥ 3); negative (relative)
//! indices; comments (`#`), blank lines, CRLF, leading/trailing whitespace.
//! Missing `vn` → flat per-face normals (identical to the glTF path); missing
//! `vt` → `(0, 0)`. Lines/points (`l`, `p`) and free-form surface statements are
//! ignored.
//!
//! Materials (`.mtl`): **only** `newmtl`, `Kd` (→ base color), and `map_Kd`
//! (→ base-color texture) are read. Everything else — `Ka`, `Ks`, `Ns`, `Ni`,
//! `d`/`Tr`, `illum`, `map_Ks`, `map_Bump`/`bump`, `norm`, displacement — is
//! ignored; metallic/roughness/emissive fall back to non-metal defaults
//! (`metallic = 0`, `roughness = 1`, `emissive = 0`). A `map_Kd` filename is the
//! last whitespace token (leading `-o/-s/-bm …` option args are dropped;
//! filenames containing spaces are not supported). A missing/unreadable `.mtl`
//! is non-fatal (the referenced slots degrade to default color); a
//! missing/unreadable `map_Kd` image degrades that material to color-only.

use std::collections::HashMap;
use std::path::Path;

use glam::Vec3;

use crate::asset::{MeshAsset, MeshVertex, SubMesh};
use crate::error::MeshError;
use crate::gltf_import::{GltfImport, ImportedMaterial, ImportedMesh, RawImage};
use crate::optimize::optimize;

/// A submesh under construction: deduplicated vertices keyed by the OBJ
/// `(v, vt, vn)` reference triple, plus the triangle index stream.
struct SubBuilder {
    material_slot: Option<u32>,
    verts: Vec<MeshVertex>,
    indices: Vec<u32>,
    /// `(v_index, vt_index, vn_index)` (each 0-based, `-1` = absent) → output
    /// vertex index. De-indexes the OBJ's independent attribute streams.
    dedup: HashMap<(i64, i64, i64), u32>,
    /// True once any face vertex in this submesh referenced a `vn` — gates flat
    /// normal generation (mirrors the glTF "normals present for the whole
    /// primitive or not at all" rule).
    had_normals: bool,
}

impl SubBuilder {
    fn new(material_slot: Option<u32>) -> Self {
        Self {
            material_slot,
            verts: Vec::new(),
            indices: Vec::new(),
            dedup: HashMap::new(),
            had_normals: false,
        }
    }
}

/// A mesh under construction (one OBJ `o` object).
struct MeshBuilder {
    name: String,
    subs: Vec<SubBuilder>,
}

/// Import a Wavefront `.obj` file into the shared [`GltfImport`] container.
///
/// Returns [`MeshError::Gltf`] (message prefixed `obj:` with a 1-based line
/// number) on a malformed line or an out-of-range index, and [`MeshError::Io`]
/// if the `.obj` itself can't be read.
pub fn import_obj(path: &Path) -> Result<GltfImport, MeshError> {
    let content = std::fs::read_to_string(path)?;
    let obj_dir = path.parent().unwrap_or_else(|| Path::new("."));

    let mut out = GltfImport::default();

    // Material tables kept in lockstep: `mat_names[i]` is the slot-`i` name,
    // `out.materials[i]` its descriptor, `mat_index[name] == i`.
    let mut mat_names: Vec<String> = Vec::new();
    let mut mat_index: HashMap<String, usize> = HashMap::new();
    // Resolved-image-path → index into `out.images` (dedupes shared textures).
    let mut image_index: HashMap<String, usize> = HashMap::new();

    // OBJ shares these attribute pools across every object.
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();

    let mut meshes: Vec<MeshBuilder> = Vec::new();
    let mut cur_mesh: Option<usize> = None;
    let mut cur_sub: Option<usize> = None;
    let mut cur_material: Option<u32> = None;

    for (lineno, raw_line) in content.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut tokens = line.split_whitespace();
        let Some(kw) = tokens.next() else { continue };
        // 1-based for human-facing errors.
        let ln = lineno + 1;

        match kw {
            "v" => {
                let p = parse_vec3(&mut tokens, ln)?;
                positions.push(p);
            }
            "vn" => {
                let n = parse_vec3(&mut tokens, ln)?;
                normals.push(n);
            }
            "vt" => {
                let u = next_f32(&mut tokens, ln)?;
                // `v` is optional in the spec (1-D textures); default 0.
                let v = tokens
                    .next()
                    .map(|t| parse_f32(t, ln))
                    .transpose()?
                    .unwrap_or(0.0);
                // Flip to match the glTF importer's upper-left UV origin.
                uvs.push([u, 1.0 - v]);
            }
            "o" => {
                let name = rest_or(line, "Mesh");
                meshes.push(MeshBuilder {
                    name,
                    subs: Vec::new(),
                });
                cur_mesh = Some(meshes.len() - 1);
                cur_sub = None;
            }
            "g" => {
                // A group boundary starts a fresh submesh in the current mesh.
                cur_sub = None;
            }
            "usemtl" => {
                let name = rest_or(line, "");
                let slot = if name.is_empty() {
                    None
                } else {
                    Some(intern_material(
                        &name,
                        &mut mat_names,
                        &mut mat_index,
                        &mut out.materials,
                    ))
                };
                cur_material = slot;
                cur_sub = None;
            }
            "mtllib" => {
                for file in tokens {
                    load_mtl(
                        obj_dir,
                        file,
                        &mut out,
                        &mut mat_names,
                        &mut mat_index,
                        &mut image_index,
                    );
                }
            }
            "f" => {
                // Resolve every face vertex to (position, uv, normal) indices.
                let mut face: Vec<(usize, Option<usize>, Option<usize>)> = Vec::new();
                for tok in tokens {
                    let (v, vt, vn) =
                        parse_face_vertex(tok, positions.len(), uvs.len(), normals.len(), ln)?;
                    face.push((v, vt, vn));
                }
                if face.len() < 3 {
                    return Err(err(ln, "face needs at least 3 vertices"));
                }

                // Ensure a current mesh + submesh exist (lazily, so empty
                // objects/groups never produce empty submeshes).
                let mi = match cur_mesh {
                    Some(mi) => mi,
                    None => {
                        meshes.push(MeshBuilder {
                            name: "Mesh".to_string(),
                            subs: Vec::new(),
                        });
                        cur_mesh = Some(meshes.len() - 1);
                        meshes.len() - 1
                    }
                };
                let si = match cur_sub {
                    Some(si) => si,
                    None => {
                        meshes[mi].subs.push(SubBuilder::new(cur_material));
                        let si = meshes[mi].subs.len() - 1;
                        cur_sub = Some(si);
                        si
                    }
                };
                let sub = &mut meshes[mi].subs[si];

                // Fan-triangulate: (0, i, i+1). OBJ winds CCW, same as glTF.
                let mut local: Vec<u32> = Vec::with_capacity(face.len());
                for &(v, vt, vn) in &face {
                    if vn.is_some() {
                        sub.had_normals = true;
                    }
                    local.push(add_vertex(sub, &positions, &normals, &uvs, v, vt, vn));
                }
                for i in 1..local.len() - 1 {
                    sub.indices.push(local[0]);
                    sub.indices.push(local[i]);
                    sub.indices.push(local[i + 1]);
                }
            }
            // Everything else (l, p, s, mtllib options already handled, vp,
            // curve/surface statements, …) is silently ignored.
            _ => {}
        }
    }

    // Finalize: generate missing normals, run the shared meshopt pass, drop
    // empty submeshes/meshes, and assemble `MeshAsset`s.
    for mb in meshes {
        let mut submeshes = Vec::new();
        for (ordinal, sb) in mb.subs.into_iter().enumerate() {
            if sb.verts.is_empty() || sb.indices.is_empty() {
                continue;
            }
            let mut verts = sb.verts;
            if !sb.had_normals {
                let flat = compute_normals(&verts, &sb.indices);
                for (vtx, n) in verts.iter_mut().zip(flat) {
                    vtx.normal = n;
                }
            }
            // Same weld → vertex-cache → vertex-fetch pass the glTF rigid path uses.
            let (verts, indices) = optimize(verts, sb.indices);
            submeshes.push(SubMesh {
                name: format!("{}_{}", mb.name, ordinal),
                vertices: verts,
                indices,
                material_slot: sb.material_slot,
                skin: Vec::new(),
            });
        }
        if submeshes.is_empty() {
            continue;
        }
        out.meshes.push(ImportedMesh {
            name: mb.name,
            mesh: MeshAsset::new(submeshes, mat_names.clone()),
            skin: None,
        });
    }

    if out.meshes.is_empty() {
        return Err(MeshError::Gltf(format!("{path:?}: no faces")));
    }
    Ok(out)
}

/// De-index one face vertex into the submesh, deduplicating by its OBJ triple.
fn add_vertex(
    sub: &mut SubBuilder,
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    uvs: &[[f32; 2]],
    v: usize,
    vt: Option<usize>,
    vn: Option<usize>,
) -> u32 {
    let key = (
        v as i64,
        vt.map(|i| i as i64).unwrap_or(-1),
        vn.map(|i| i as i64).unwrap_or(-1),
    );
    if let Some(&idx) = sub.dedup.get(&key) {
        return idx;
    }
    let vertex = MeshVertex {
        position: positions[v],
        normal: vn.map(|i| normals[i]).unwrap_or([0.0, 1.0, 0.0]),
        uv: vt.map(|i| uvs[i]).unwrap_or([0.0, 0.0]),
        tangent: crate::TANGENT_PLACEHOLDER,
    };
    let idx = sub.verts.len() as u32;
    sub.verts.push(vertex);
    sub.dedup.insert(key, idx);
    idx
}

/// Parse one `f` reference token (`v`, `v/vt`, `v//vn`, `v/vt/vn`) into 0-based
/// indices, resolving negative (relative) indices against the current counts.
fn parse_face_vertex(
    tok: &str,
    n_pos: usize,
    n_uv: usize,
    n_norm: usize,
    ln: usize,
) -> Result<(usize, Option<usize>, Option<usize>), MeshError> {
    let mut parts = tok.split('/');
    let v_raw = parts
        .next()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| err(ln, "face vertex missing position index"))?;
    let v = resolve_index(v_raw, n_pos, ln, "vertex")?;

    let vt = match parts.next() {
        Some(s) if !s.is_empty() => Some(resolve_index(s, n_uv, ln, "texcoord")?),
        _ => None,
    };
    let vn = match parts.next() {
        Some(s) if !s.is_empty() => Some(resolve_index(s, n_norm, ln, "normal")?),
        _ => None,
    };
    Ok((v, vt, vn))
}

/// Resolve a 1-based (positive) or relative (negative) OBJ index to a 0-based
/// index, erroring if it is `0` or out of range.
fn resolve_index(s: &str, count: usize, ln: usize, what: &str) -> Result<usize, MeshError> {
    let raw: i64 = s
        .parse()
        .map_err(|_| err(ln, &format!("invalid {what} index {s:?}")))?;
    let idx = if raw > 0 {
        raw - 1
    } else if raw < 0 {
        count as i64 + raw
    } else {
        return Err(err(
            ln,
            &format!("{what} index 0 is invalid (OBJ is 1-based)"),
        ));
    };
    if idx < 0 || idx as usize >= count {
        return Err(err(
            ln,
            &format!("{what} index {raw} out of range (have {count})"),
        ));
    }
    Ok(idx as usize)
}

/// Read the first three floats of a `v`/`vn` line (extra components ignored).
fn parse_vec3<'a>(
    tokens: &mut impl Iterator<Item = &'a str>,
    ln: usize,
) -> Result<[f32; 3], MeshError> {
    let x = next_f32(tokens, ln)?;
    let y = next_f32(tokens, ln)?;
    let z = next_f32(tokens, ln)?;
    Ok([x, y, z])
}

fn next_f32<'a>(tokens: &mut impl Iterator<Item = &'a str>, ln: usize) -> Result<f32, MeshError> {
    let tok = tokens.next().ok_or_else(|| err(ln, "expected a number"))?;
    parse_f32(tok, ln)
}

fn parse_f32(tok: &str, ln: usize) -> Result<f32, MeshError> {
    tok.parse::<f32>()
        .map_err(|_| err(ln, &format!("invalid number {tok:?}")))
}

/// Everything after the keyword on `line`, trimmed; `default` if empty.
fn rest_or(line: &str, default: &str) -> String {
    match line.split_once(char::is_whitespace) {
        Some((_, rest)) => {
            let rest = rest.trim();
            if rest.is_empty() {
                default.to_string()
            } else {
                rest.to_string()
            }
        }
        None => default.to_string(),
    }
}

/// Intern a material name into the shared slot list, creating a default-color
/// descriptor on first sight; returns its slot index.
fn intern_material(
    name: &str,
    mat_names: &mut Vec<String>,
    mat_index: &mut HashMap<String, usize>,
    materials: &mut Vec<ImportedMaterial>,
) -> u32 {
    if let Some(&i) = mat_index.get(name) {
        return i as u32;
    }
    let i = mat_names.len();
    mat_names.push(name.to_string());
    materials.push(ImportedMaterial {
        name: name.to_string(),
        base_color: [1.0, 1.0, 1.0, 1.0],
        metallic: 0.0,
        roughness: 1.0,
        emissive: [0.0, 0.0, 0.0],
        base_color_image: None,
        normal_image: None,
        metallic_roughness_image: None,
    });
    mat_index.insert(name.to_string(), i);
    i as u32
}

/// Parse a `.mtl` sidecar for `newmtl`/`Kd`/`map_Kd` only. Missing/unreadable
/// files are non-fatal (returns having done nothing).
fn load_mtl(
    obj_dir: &Path,
    file: &str,
    out: &mut GltfImport,
    mat_names: &mut Vec<String>,
    mat_index: &mut HashMap<String, usize>,
    image_index: &mut HashMap<String, usize>,
) {
    let mtl_path = obj_dir.join(file);
    let Ok(content) = std::fs::read_to_string(&mtl_path) else {
        return; // degrade: slots resolve to default color via `usemtl`.
    };
    let mtl_dir = mtl_path.parent().unwrap_or(obj_dir).to_path_buf();

    let mut cur: Option<usize> = None;
    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut tokens = line.split_whitespace();
        let Some(kw) = tokens.next() else { continue };
        match kw {
            "newmtl" => {
                let name = rest_or(line, "Material");
                let slot = intern_material(&name, mat_names, mat_index, &mut out.materials);
                cur = Some(slot as usize);
            }
            "Kd" => {
                if let Some(i) = cur {
                    let r = tokens.next().and_then(|t| t.parse().ok()).unwrap_or(1.0);
                    let g = tokens.next().and_then(|t| t.parse().ok()).unwrap_or(1.0);
                    let b = tokens.next().and_then(|t| t.parse().ok()).unwrap_or(1.0);
                    out.materials[i].base_color = [r, g, b, 1.0];
                }
            }
            "map_Kd" => {
                if let Some(i) = cur {
                    // Filename is the last token (drops leading `-o/-s/…` args).
                    if let Some(fname) = line.split_whitespace().last() {
                        if let Some(img_idx) = load_image(&mtl_dir, fname, out, image_index) {
                            out.materials[i].base_color_image = Some(img_idx);
                        }
                    }
                }
            }
            _ => {} // Ka/Ks/Ns/Ni/d/illum/map_Ks/bump/norm/… ignored.
        }
    }
}

/// Decode a `map_Kd` image to an RGBA8 [`RawImage`], deduped by resolved path.
/// Returns `None` (color-only degrade) if the file is missing or undecodable.
fn load_image(
    dir: &Path,
    fname: &str,
    out: &mut GltfImport,
    image_index: &mut HashMap<String, usize>,
) -> Option<usize> {
    let path = dir.join(fname);
    let key = path.to_string_lossy().to_string();
    if let Some(&i) = image_index.get(&key) {
        return Some(i);
    }
    let bytes = std::fs::read(&path).ok()?;
    let img = image::load_from_memory(&bytes).ok()?.to_rgba8();
    let (width, height) = img.dimensions();
    let name = Path::new(fname)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("texture")
        .to_string();
    let idx = out.images.len();
    out.images.push(RawImage {
        name,
        width,
        height,
        rgba8: img.into_raw(),
    });
    image_index.insert(key, idx);
    Some(idx)
}

/// Flat per-face normals for geometry that ships without them. Identical
/// algorithm to the glTF importer's `compute_normals` (kept local because that
/// module's helper is private).
fn compute_normals(verts: &[MeshVertex], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut acc = vec![Vec3::ZERO; verts.len()];
    for tri in indices.chunks_exact(3) {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let pa = Vec3::from(verts[a].position);
        let pb = Vec3::from(verts[b].position);
        let pc = Vec3::from(verts[c].position);
        let n = (pb - pa).cross(pc - pa);
        acc[a] += n;
        acc[b] += n;
        acc[c] += n;
    }
    acc.into_iter()
        .map(|v| v.normalize_or_zero().to_array())
        .collect()
}

fn err(ln: usize, msg: &str) -> MeshError {
    MeshError::Gltf(format!("obj: line {ln}: {msg}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Import OBJ text written to a temp dir (so `mtllib`/`map_Kd` relative
    /// paths resolve). Returns the parsed container.
    fn import_str(obj: &str) -> Result<GltfImport, MeshError> {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.obj");
        std::fs::write(&path, obj).unwrap();
        import_obj(&path)
    }

    #[test]
    fn minimal_triangle() {
        let g = import_str("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").unwrap();
        assert_eq!(g.meshes.len(), 1);
        let m = &g.meshes[0].mesh;
        assert_eq!(m.triangle_count(), 1);
        assert_eq!(m.vertex_count(), 3);
        // No skeletons/clips ever come out of OBJ.
        assert!(g.skeletons.is_empty() && g.clips.is_empty());
        // Normals were generated: the CCW triangle in the XY plane faces +Z.
        for v in &m.submeshes[0].vertices {
            assert!(
                v.normal[2] > 0.9,
                "generated normal faces +Z, got {:?}",
                v.normal
            );
        }
    }

    #[test]
    fn quad_fans_into_two_triangles_with_ccw_winding() {
        // A single 4-gon face → fan of 2 triangles.
        let g = import_str("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n").unwrap();
        let m = &g.meshes[0].mesh;
        assert_eq!(m.triangle_count(), 2);
        assert_eq!(m.vertex_count(), 4, "four unique corners");
        // CCW winding preserved → all flat normals point +Z.
        for v in &m.submeshes[0].vertices {
            assert!(v.normal[2] > 0.9, "CCW quad faces +Z, got {:?}", v.normal);
        }
    }

    #[test]
    fn v_slash_slash_vn_uses_supplied_normals() {
        // `v//vn` form: normals present, no texcoords. Down-facing normal must
        // survive (NOT be overwritten by flat generation).
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 -1\nf 1//1 2//1 3//1\n";
        let g = import_str(obj).unwrap();
        let m = &g.meshes[0].mesh;
        for v in &m.submeshes[0].vertices {
            assert_eq!(v.normal, [0.0, 0.0, -1.0]);
            assert_eq!(v.uv, [0.0, 0.0], "no vt → default UV");
        }
    }

    #[test]
    fn uv_v_axis_is_flipped_to_gltf_convention() {
        let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0.25 0.75\nf 1/1 2/1 3/1\n";
        let g = import_str(obj).unwrap();
        let uv = g.meshes[0].mesh.submeshes[0].vertices[0].uv;
        assert_eq!(uv, [0.25, 0.25], "v flipped: 1 - 0.75 = 0.25");
    }

    #[test]
    fn negative_relative_indices() {
        // -1/-2/-3 reference the three most-recently-declared vertices.
        let g = import_str("v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n").unwrap();
        let m = &g.meshes[0].mesh;
        assert_eq!(m.triangle_count(), 1);
        assert_eq!(m.vertex_count(), 3);
    }

    #[test]
    fn two_objects_split_into_two_meshes() {
        let obj = "\
o first
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
o second
v 0 0 1
v 1 0 1
v 0 1 1
f 4 5 6
";
        let g = import_str(obj).unwrap();
        assert_eq!(g.meshes.len(), 2, "two `o` objects → two meshes");
        assert_eq!(g.meshes[0].name, "first");
        assert_eq!(g.meshes[1].name, "second");
        assert_eq!(g.meshes[0].mesh.triangle_count(), 1);
        assert_eq!(g.meshes[1].mesh.triangle_count(), 1);
    }

    #[test]
    fn usemtl_splits_submeshes_and_wires_mtl_color_and_texture() {
        let dir = tempfile::tempdir().unwrap();
        // A tiny 2×2 RGBA PNG for `map_Kd`.
        let px: Vec<u8> = vec![255; 2 * 2 * 4];
        image::RgbaImage::from_raw(2, 2, px)
            .unwrap()
            .save(dir.path().join("albedo.png"))
            .unwrap();
        std::fs::write(
            dir.path().join("m.mtl"),
            "newmtl red\nKd 1 0 0\nnewmtl tex\nKd 0 1 0\nmap_Kd albedo.png\n",
        )
        .unwrap();
        let obj = "\
mtllib m.mtl
v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
usemtl red
f 1 2 3
usemtl tex
f 1 2 4
";
        std::fs::write(dir.path().join("m.obj"), obj).unwrap();
        let g = import_obj(&dir.path().join("m.obj")).unwrap();

        // One mesh, two submeshes (one per usemtl run).
        assert_eq!(g.meshes.len(), 1);
        let subs = &g.meshes[0].mesh.submeshes;
        assert_eq!(subs.len(), 2, "usemtl boundary splits submeshes");

        // Two materials in the shared slot list; `red` Kd landed as base color.
        assert_eq!(g.materials.len(), 2);
        let red = g.materials.iter().find(|m| m.name == "red").unwrap();
        assert_eq!(red.base_color, [1.0, 0.0, 0.0, 1.0]);
        assert!(red.base_color_image.is_none());

        // `tex` wired its map_Kd to a decoded image entry.
        let tex = g.materials.iter().find(|m| m.name == "tex").unwrap();
        assert_eq!(tex.base_color, [0.0, 1.0, 0.0, 1.0]);
        let img = tex.base_color_image.expect("map_Kd wired");
        assert_eq!(g.images[img].width, 2);
        assert_eq!(g.images[img].height, 2);
        assert_eq!(g.images[img].rgba8.len(), 2 * 2 * 4);

        // Submesh material slots index the shared list correctly.
        let slot_names: Vec<&str> = subs
            .iter()
            .map(|s| g.meshes[0].mesh.material_slots[s.material_slot.unwrap() as usize].as_str())
            .collect();
        assert!(slot_names.contains(&"red") && slot_names.contains(&"tex"));
    }

    #[test]
    fn missing_map_kd_degrades_to_color_only() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("m.mtl"),
            "newmtl m\nKd 0.5 0.5 0.5\nmap_Kd nonexistent.png\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("m.obj"),
            "mtllib m.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl m\nf 1 2 3\n",
        )
        .unwrap();
        let g = import_obj(&dir.path().join("m.obj")).unwrap();
        assert_eq!(g.images.len(), 0, "unreadable map_Kd → no image");
        assert_eq!(g.materials[0].base_color, [0.5, 0.5, 0.5, 1.0]);
        assert!(g.materials[0].base_color_image.is_none());
    }

    #[test]
    fn crlf_comments_and_blank_lines() {
        let obj = "# header comment\r\n\r\nv 0 0 0\r\nv 1 0 0\r\n  # indented comment\r\nv 0 1 0\r\n\r\nf 1 2 3\r\n";
        let g = import_str(obj).unwrap();
        assert_eq!(g.meshes.len(), 1);
        assert_eq!(g.meshes[0].mesh.triangle_count(), 1);
    }

    #[test]
    fn out_of_range_index_errors_with_line_number() {
        // `f` on line 4 references vertex 9 (only 3 exist).
        let err = import_str("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9\n").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("line 4"), "reports the offending line: {msg}");
        assert!(msg.contains("out of range"), "explains the failure: {msg}");
    }

    #[test]
    fn empty_geometry_errors() {
        assert!(import_str("# just a comment\n").is_err());
    }
}
