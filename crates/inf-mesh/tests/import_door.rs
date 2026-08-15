//! **Permanent refusal fixtures for the import door** (hardening wave B, root
//! unit U1).
//!
//! Every test here hand-builds a *malformed* source document — a real `.gltf`
//! with a real binary buffer, or a real `.obj` — and asserts that
//! [`inf_mesh::import_gltf`] / [`inf_mesh::import_obj`] **refuse** it by name.
//! They are fixtures rather than unit tests on purpose: the unit tests in
//! `inf_mesh::validate` prove the predicates, and these prove the predicates are
//! *wired to the doors*, which is the half that a refactor silently loses.
//!
//! What each one would do if the door were removed is recorded in its own doc
//! comment, because "this file is refused" is not a claim anybody can check
//! against a consequence.
//!
//! The mutation that verifies this suite: delete the matching `reject_*` call
//! from `gltf_import.rs` / the `is_finite` check from `obj_import::parse_f32`
//! and **exactly one** test fails, the one that names that check. Measured, on
//! three of them:
//!
//! * `reject_out_of_range` removed → `an_index_past_the_vertex_buffer_is_refused`
//!   alone fails, in debug by panicking inside `compute_normals` (in release the
//!   same input reaches meshopt's FFI instead, which is the finding);
//! * `parse_f32`'s finiteness check removed →
//!   `every_spelling_of_a_non_finite_obj_number_is_refused` alone fails, by
//!   **importing successfully**;
//! * the `WEIGHTS_0` refusal *and* `VertexSkin::normalized`'s `is_finite` both
//!   reverted → `an_infinite_skin_weight_is_refused` alone fails, by importing
//!   successfully. Reverting only one of the two leaves the fixture green, which
//!   is the point of fixing both.
#![cfg(not(target_arch = "wasm32"))]

use std::path::{Path, PathBuf};

use inf_mesh::{import_gltf, import_obj, MeshError};

// ── fixture plumbing ────────────────────────────────────────────────────────

/// Append `f32`s to `buf`, returning the byte offset they start at.
fn f32s(buf: &mut Vec<u8>, vals: &[f32]) -> usize {
    let off = buf.len();
    for v in vals {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    off
}

/// Append `u16`s to `buf` (4-byte aligned before and after), returning the
/// offset.
fn u16s(buf: &mut Vec<u8>, vals: &[u16]) -> usize {
    while !buf.len().is_multiple_of(4) {
        buf.push(0);
    }
    let off = buf.len();
    for v in vals {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    while !buf.len().is_multiple_of(4) {
        buf.push(0);
    }
    off
}

/// Write a `.gltf` + its `.bin` into `dir` and return the `.gltf` path.
fn write_gltf(dir: &Path, json: String, bin: &[u8]) -> PathBuf {
    std::fs::write(dir.join("f.bin"), bin).unwrap();
    let path = dir.join("f.gltf");
    std::fs::write(&path, json).unwrap();
    path
}

/// Import a `.gltf` built from `json` + `bin`, keeping the temp dir alive for
/// the call.
fn import(json: String, bin: Vec<u8>) -> Result<(), MeshError> {
    let dir = tempfile::tempdir().unwrap();
    let path = write_gltf(dir.path(), json, &bin);
    import_gltf(&path).map(|_| ())
}

/// Assert an import was refused and that the refusal *names* the problem — a
/// refusal nobody can act on is the same as no message.
#[track_caller]
fn refused(res: Result<(), MeshError>, needle: &str) {
    match res {
        Ok(()) => panic!("the import door accepted a malformed document ({needle})"),
        Err(e) => {
            let msg = e.to_string();
            assert!(
                msg.contains(needle),
                "refused, but the message does not name the problem\n  wanted: {needle}\n  got:    {msg}"
            );
        }
    }
}

/// A one-triangle glTF whose POSITION values and index values are given, so a
/// test can poison exactly one of them.
fn triangle(positions: &[f32], indices: &[u16]) -> (String, Vec<u8>) {
    let mut buf = Vec::new();
    let pos_off = f32s(&mut buf, positions);
    let pos_len = positions.len() * 4;
    let idx_off = u16s(&mut buf, indices);
    let json = format!(
        r#"{{
  "asset": {{ "version": "2.0" }},
  "scene": 0,
  "scenes": [{{ "nodes": [0] }}],
  "nodes": [{{ "mesh": 0 }}],
  "meshes": [{{ "name": "Tri", "primitives": [{{ "attributes": {{ "POSITION": 0 }}, "indices": 1 }}] }}],
  "buffers": [{{ "uri": "f.bin", "byteLength": {total} }}],
  "bufferViews": [
    {{ "buffer": 0, "byteOffset": {pos_off}, "byteLength": {pos_len} }},
    {{ "buffer": 0, "byteOffset": {idx_off}, "byteLength": {idx_len} }}
  ],
  "accessors": [
    {{ "bufferView": 0, "componentType": 5126, "count": {pos_count}, "type": "VEC3",
       "min": [0,0,0], "max": [1,1,0] }},
    {{ "bufferView": 1, "componentType": 5123, "count": {idx_count}, "type": "SCALAR" }}
  ]
}}"#,
        total = buf.len(),
        pos_count = positions.len() / 3,
        idx_len = indices.len() * 2,
        idx_count = indices.len(),
    );
    (json, buf)
}

const TRI: [f32; 9] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

// ── glTF: indices ───────────────────────────────────────────────────────────

/// **C4-1 / C4-9 — an index past the end of the vertex buffer.**
///
/// Without the door this reaches `meshopt::generate_vertex_remap`, a raw
/// `unsafe` FFI call whose remap table is sized `vertices.len()` and whose C
/// side writes `remap[index]` with its assert compiled out in release: an
/// out-of-bounds heap **write**, not a panic.
#[test]
fn an_index_past_the_vertex_buffer_is_refused() {
    let (json, bin) = triangle(&TRI, &[0, 1, 7]);
    refused(import(json, bin), "addresses outside a 3-vertex buffer");
}

// ── glTF: finiteness ────────────────────────────────────────────────────────

/// **C4-8 — a NaN in POSITION.**
///
/// glTF accessor floats come out of the binary buffer with any bit pattern; no
/// parser filters anything. Without the door the NaN lands in `.inf_mesh` and
/// then *hides*: `Aabb::grow` folds with `f32::min`/`max`, which ignore NaN, so
/// the bounds look healthy over a poisoned vertex buffer, `MeshAsset`'s derived
/// `PartialEq` breaks every save→load→save gate, and the content-hash dedupe
/// index sees two spellings of NaN as two different assets.
#[test]
fn a_nan_position_is_refused() {
    let mut poisoned = TRI;
    poisoned[4] = f32::NAN;
    let (json, bin) = triangle(&poisoned, &[0, 1, 2]);
    refused(import(json, bin), "not a finite number");
}

/// **C4-8 — an infinity in POSITION**, which the same fold hides just as well
/// and which additionally makes `inf_vgeom`'s `bounding_sphere` produce an
/// infinite centre with a **zero radius** (`inf - inf`), a cull bound the
/// GPU-parity gate compares.
#[test]
fn an_infinite_position_is_refused() {
    let mut poisoned = TRI;
    poisoned[0] = f32::INFINITY;
    let (json, bin) = triangle(&poisoned, &[0, 1, 2]);
    refused(import(json, bin), "not a finite number");
}

// ── glTF: stream-length agreement ───────────────────────────────────────────

/// **C4-6 — a `TEXCOORD_0` accessor shorter than `POSITION`.**
///
/// The vertex-assembly loop indexed `uvs[i]` raw while the `normals` line
/// directly above it used `.get(i).unwrap_or(..)`; the asymmetry was visible in
/// four adjacent lines. A short accessor panicked the importer.
#[test]
fn a_short_texcoord_accessor_is_refused() {
    let mut buf = Vec::new();
    let pos_off = f32s(&mut buf, &TRI);
    // Two uvs for three positions.
    let uv_off = f32s(&mut buf, &[0.0, 0.0, 1.0, 0.0]);
    let idx_off = u16s(&mut buf, &[0, 1, 2]);
    let json = format!(
        r#"{{
  "asset": {{ "version": "2.0" }},
  "scene": 0,
  "scenes": [{{ "nodes": [0] }}],
  "nodes": [{{ "mesh": 0 }}],
  "meshes": [{{ "name": "Tri", "primitives": [{{
    "attributes": {{ "POSITION": 0, "TEXCOORD_0": 1 }}, "indices": 2
  }}] }}],
  "buffers": [{{ "uri": "f.bin", "byteLength": {total} }}],
  "bufferViews": [
    {{ "buffer": 0, "byteOffset": {pos_off}, "byteLength": 36 }},
    {{ "buffer": 0, "byteOffset": {uv_off}, "byteLength": 16 }},
    {{ "buffer": 0, "byteOffset": {idx_off}, "byteLength": 6 }}
  ],
  "accessors": [
    {{ "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
       "min": [0,0,0], "max": [1,1,0] }},
    {{ "bufferView": 1, "componentType": 5126, "count": 2, "type": "VEC2" }},
    {{ "bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR" }}
  ]
}}"#,
        total = buf.len(),
    );
    refused(import(json, buf), "2 elements against 3");
}

/// **C4-37 — `+inf` in `WEIGHTS_0`.**
///
/// `VertexSkin::normalized`'s `sum > 1e-6` correctly rejects NaN and *passes*
/// `+inf`, so `inf / inf` manufactures a NaN out of an input that never held
/// one — and that NaN is `Pod` and goes straight to a GPU vertex buffer.
#[test]
fn an_infinite_skin_weight_is_refused() {
    let mut buf = Vec::new();
    let pos_off = f32s(&mut buf, &TRI);
    let wgt_off = f32s(
        &mut buf,
        &[
            1.0,
            0.0,
            0.0,
            0.0,
            f32::INFINITY,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ],
    );
    let idx_off = u16s(&mut buf, &[0, 1, 2]);
    let json = format!(
        r#"{{
  "asset": {{ "version": "2.0" }},
  "scene": 0,
  "scenes": [{{ "nodes": [0] }}],
  "nodes": [{{ "mesh": 0 }}],
  "meshes": [{{ "name": "Tri", "primitives": [{{
    "attributes": {{ "POSITION": 0, "WEIGHTS_0": 1 }}, "indices": 2
  }}] }}],
  "buffers": [{{ "uri": "f.bin", "byteLength": {total} }}],
  "bufferViews": [
    {{ "buffer": 0, "byteOffset": {pos_off}, "byteLength": 36 }},
    {{ "buffer": 0, "byteOffset": {wgt_off}, "byteLength": 48 }},
    {{ "buffer": 0, "byteOffset": {idx_off}, "byteLength": 6 }}
  ],
  "accessors": [
    {{ "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
       "min": [0,0,0], "max": [1,1,0] }},
    {{ "bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC4" }},
    {{ "bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR" }}
  ]
}}"#,
        total = buf.len(),
    );
    refused(import(json, buf), "WEIGHTS_0");
}

// ── glTF: the skinned document ──────────────────────────────────────────────

/// A three-joint skinned arm whose `JOINTS_0` values and animation sampler are
/// caller-supplied, so one test can poison exactly one of them. Mirrors the
/// healthy fixture in `gltf_import`'s own tests.
fn skinned_arm(joints: &[u16; 12], times: &[f32], rotations: &[f32]) -> (String, Vec<u8>) {
    let mut buf = Vec::new();
    let pos_off = f32s(&mut buf, &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0]);
    let wgt_off = f32s(
        &mut buf,
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    );
    let ibm_off = {
        let off = buf.len();
        for _ in 0..3 {
            for v in glam::Mat4::IDENTITY.to_cols_array() {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        off
    };
    let time_off = f32s(&mut buf, times);
    let rot_off = f32s(&mut buf, rotations);
    let idx_off = u16s(&mut buf, &[0, 1, 2]);
    let joint_off = u16s(&mut buf, joints);
    let json = format!(
        r#"{{
  "asset": {{ "version": "2.0" }},
  "scene": 0,
  "scenes": [{{ "nodes": [0, 3] }}],
  "nodes": [
    {{ "name": "joint0", "translation": [0,0,0], "children": [1] }},
    {{ "name": "joint1", "translation": [0,1,0], "children": [2] }},
    {{ "name": "joint2", "translation": [0,1,0] }},
    {{ "name": "armMeshNode", "mesh": 0, "skin": 0 }}
  ],
  "meshes": [{{ "name": "Arm", "primitives": [{{
    "attributes": {{ "POSITION": 0, "JOINTS_0": 6, "WEIGHTS_0": 1 }},
    "indices": 5
  }}] }}],
  "skins": [{{ "name": "ArmSkel", "joints": [0,1,2], "inverseBindMatrices": 2 }}],
  "animations": [{{
    "name": "Wave",
    "channels": [{{ "sampler": 0, "target": {{ "node": 1, "path": "rotation" }} }}],
    "samplers": [{{ "input": 3, "output": 4, "interpolation": "LINEAR" }}]
  }}],
  "buffers": [{{ "uri": "f.bin", "byteLength": {total} }}],
  "bufferViews": [
    {{ "buffer": 0, "byteOffset": {pos_off}, "byteLength": 36 }},
    {{ "buffer": 0, "byteOffset": {wgt_off}, "byteLength": 48 }},
    {{ "buffer": 0, "byteOffset": {ibm_off}, "byteLength": 192 }},
    {{ "buffer": 0, "byteOffset": {time_off}, "byteLength": {time_len} }},
    {{ "buffer": 0, "byteOffset": {rot_off}, "byteLength": {rot_len} }},
    {{ "buffer": 0, "byteOffset": {idx_off}, "byteLength": 6 }},
    {{ "buffer": 0, "byteOffset": {joint_off}, "byteLength": 24 }}
  ],
  "accessors": [
    {{ "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
       "min": [0,0,0], "max": [1,2,0] }},
    {{ "bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC4" }},
    {{ "bufferView": 2, "componentType": 5126, "count": 3, "type": "MAT4" }},
    {{ "bufferView": 3, "componentType": 5126, "count": {time_count}, "type": "SCALAR",
       "min": [0.0], "max": [1.0] }},
    {{ "bufferView": 4, "componentType": 5126, "count": {rot_count}, "type": "VEC4" }},
    {{ "bufferView": 5, "componentType": 5123, "count": 3, "type": "SCALAR" }},
    {{ "bufferView": 6, "componentType": 5123, "count": 3, "type": "VEC4" }}
  ]
}}"#,
        total = buf.len(),
        time_len = times.len() * 4,
        time_count = times.len(),
        rot_len = rotations.len() * 4,
        rot_count = rotations.len() / 4,
    );
    (json, buf)
}

const GOOD_JOINTS: [u16; 12] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0];
const TWO_KEYS: [f32; 2] = [0.0, 1.0];
const TWO_ROTS: [f32; 8] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.707_106_77, 0.707_106_77];

/// The control: the same document, unpoisoned, still imports. Without this the
/// refusal fixtures below cannot distinguish "the door caught it" from "the
/// fixture never worked".
#[test]
fn the_healthy_skinned_arm_still_imports() {
    let (json, bin) = skinned_arm(&GOOD_JOINTS, &TWO_KEYS, &TWO_ROTS);
    let dir = tempfile::tempdir().unwrap();
    let path = write_gltf(dir.path(), json, &bin);
    let g = import_gltf(&path).expect("the healthy fixture must import");
    assert_eq!(g.skeletons.len(), 1);
    assert_eq!(g.clips.len(), 1);
    assert_eq!(g.meshes[0].skin, Some(0));
}

/// **C4-20 — a `JOINTS_0` value naming a joint the skin does not have.**
///
/// Nothing panics on this: the GPU path is naga-bounds-checked and the CPU path
/// clamps. The symptom is a silently wrong deformation, which is worse than a
/// refusal precisely because it looks like the author's modelling mistake.
#[test]
fn a_joint_index_past_the_skin_is_refused() {
    let mut joints = GOOD_JOINTS;
    joints[8] = 3; // the skin has joints 0..=2
    let (json, bin) = skinned_arm(&joints, &TWO_KEYS, &TWO_ROTS);
    refused(import(json, bin), "weighted to joint 3");
}

/// **C4-7 — a NaN keyframe time.**
///
/// `inf_anim::clip::locate` binary-searches `times`. A NaN makes *both* of its
/// range guards false (`NaN <= x` and `NaN >= x` are each false),
/// `partition_point` returns 0, and `let i0 = hi - 1` underflows — a panic in
/// the shipped player, at playback, long after the import that caused it.
#[test]
fn a_nan_keyframe_time_is_refused() {
    let (json, bin) = skinned_arm(&GOOD_JOINTS, &[0.0, f32::NAN], &TWO_ROTS);
    refused(import(json, bin), "not a finite number");
}

/// **C4-7 — keyframe times that do not increase.**
///
/// glTF requires strictly increasing sampler input. An unsorted list makes the
/// binary search return the wrong bracket silently; equal adjacent times are a
/// zero-length span, and a span is a divisor.
#[test]
fn non_increasing_keyframe_times_are_refused() {
    let (json, bin) = skinned_arm(&GOOD_JOINTS, &[1.0, 0.5], &TWO_ROTS);
    refused(import(json, bin), "does not increase past");
}

/// **C4-7 — a sampler whose output count differs from its input count.**
///
/// `Vec3Track::new` / `QuatTrack::new` only `debug_assert` the agreement, which
/// is compiled out in release, and `sample()` then indexes `self.values[i0]`
/// with an index derived from `times`.
#[test]
fn a_sampler_output_shorter_than_its_input_is_refused() {
    // Three input times, two output rotations.
    let (json, bin) = skinned_arm(&GOOD_JOINTS, &[0.0, 0.5, 1.0], &TWO_ROTS);
    refused(import(json, bin), "2 elements against 3");
}

/// **C4-8 — a NaN in a rotation track**, which rides `Pose.locals[].rotation`
/// into `inf_ecs::pose::pose_state_bytes`: the committed trace the
/// PIE-==-shipping parity gate compares.
#[test]
fn a_nan_track_value_is_refused() {
    let mut rots = TWO_ROTS;
    rots[5] = f32::NAN;
    let (json, bin) = skinned_arm(&GOOD_JOINTS, &TWO_KEYS, &rots);
    refused(import(json, bin), "not a finite number");
}

// ── OBJ ─────────────────────────────────────────────────────────────────────

fn obj(text: &str) -> Result<inf_mesh::GltfImport, MeshError> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("m.obj");
    std::fs::write(&path, text).unwrap();
    import_obj(&path)
}

/// **C4-8 — `str::parse::<f32>` accepts `"nan"`, `"inf"` and `"infinity"`**, so
/// every one of these is a *syntactically valid* OBJ vertex line and the format
/// has no rule against it. `parse_f32` is the single door all of `v`, `vn` and
/// `vt` come through, which is why one check covers all three.
#[test]
fn every_spelling_of_a_non_finite_obj_number_is_refused() {
    for spelling in ["nan", "NaN", "inf", "-inf", "infinity", "Infinity"] {
        let text = format!("v {spelling} 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n");
        refused(obj(&text).map(|_| ()), "is not a finite number");
        // The same door, on a normal and on a texcoord.
        let text = format!("v 0 0 0\nv 1 0 0\nv 0 1 0\nvn {spelling} 0 1\nf 1//1 2//1 3//1\n");
        refused(obj(&text).map(|_| ()), "is not a finite number");
        let text = format!("v 0 0 0\nv 1 0 0\nv 0 1 0\nvt {spelling} 0\nf 1/1 2/1 3/1\n");
        refused(obj(&text).map(|_| ()), "is not a finite number");
    }
}

/// A `.mtl` is a degrade-friendly sidecar — a missing one is skipped outright —
/// so `Kd nan nan nan` takes the documented white default rather than refusing
/// the whole model. What it must **not** do is write the NaN into `.inf_mat`.
#[test]
fn a_non_finite_diffuse_colour_falls_back_to_white_rather_than_shipping_a_nan() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("m.mtl"), "newmtl Bad\nKd nan inf 0.5\n").unwrap();
    let path = dir.path().join("m.obj");
    std::fs::write(
        &path,
        "mtllib m.mtl\nusemtl Bad\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
    )
    .unwrap();
    let g = import_obj(&path).unwrap();
    let c = g.materials[0].base_color;
    assert!(
        c.iter().all(|v| v.is_finite()),
        "a NaN reached an ImportedMaterial: {c:?}"
    );
    assert_eq!(c, [1.0, 1.0, 0.5, 1.0], "only the bad components defaulted");
}

/// The healthy OBJ control, so the refusals above are not passing because the
/// importer refuses everything.
#[test]
fn a_healthy_obj_still_imports() {
    let g = obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n").unwrap();
    assert_eq!(g.meshes[0].mesh.triangle_count(), 1);
}

// ── Round-2 finding B2: the door has a second entrance ──────────────────────
//
// Everything above is about `.gltf` / `.glb` / `.obj`. But a `.inf_mesh` **on
// disk** is also bytes somebody else wrote — the Content Drawer scans every
// loose payload under the project root, an asset arrives in a zip, a pack gets
// hand-edited — and `MeshAsset` used the DEFAULT `AssetPayload::migrate`, which
// reads one integer and asks nothing about the buffers behind it.
//
// Both production consumers of a decoded `.inf_mesh` hand its index buffer to
// `meshopt::generate_vertex_remap` through raw FFI (`inf_editor_core::assets::
// vmesh` and the cook's `build_vgeom`). These arms drive `encode` → `decode`,
// which is the exact path those consumers take.

/// A `.inf_mesh` whose index buffer addresses outside its own vertices is
/// refused at decode — C4-1's out-of-bounds heap write, at the other door.
#[test]
fn a_decoded_inf_mesh_with_a_dangling_index_is_refused() {
    use inf_mesh::{MeshAsset, MeshVertex, SubMesh};

    let sub = |indices: Vec<u32>| SubMesh {
        name: "s".into(),
        vertices: vec![MeshVertex::default(); 3],
        indices,
        material_slot: None,
        skin: Vec::new(),
    };

    // Control: the same payload with a valid index buffer round-trips.
    let good = MeshAsset::new(vec![sub(vec![0, 1, 2])], Vec::new());
    let bytes = inf_asset::encode(&good).unwrap();
    inf_asset::decode::<MeshAsset>(&bytes).expect("a healthy mesh must still decode");

    let bad = MeshAsset::new(vec![sub(vec![0, 1, 3])], Vec::new());
    let bytes = inf_asset::encode(&bad).unwrap();
    let e = inf_asset::decode::<MeshAsset>(&bytes)
        .expect_err("an index past the vertex buffer decoded as a valid mesh");
    let msg = e.to_string();
    assert!(
        msg.contains("index 3") && msg.contains("submesh 0"),
        "the refusal must name the offending index and submesh: {msg}"
    );
}

/// A NaN in a decoded `.inf_mesh` is refused. `MeshVertex` is `Pod` and is
/// uploaded verbatim, and `f32::min`/`max` ignore NaN — so the bounding box
/// computed from a poisoned buffer looks perfectly healthy and nothing
/// downstream is looking.
#[test]
fn a_decoded_inf_mesh_carrying_a_nan_is_refused() {
    use inf_mesh::{MeshAsset, MeshVertex, SubMesh};

    for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut v = vec![MeshVertex::default(); 3];
        v[1].position[2] = poison;
        let m = MeshAsset::new(
            vec![SubMesh {
                name: "s".into(),
                vertices: v,
                indices: vec![0, 1, 2],
                material_slot: None,
                skin: Vec::new(),
            }],
            Vec::new(),
        );
        let bytes = inf_asset::encode(&m).unwrap();
        assert!(
            inf_asset::decode::<MeshAsset>(&bytes).is_err(),
            "a {poison} vertex position decoded as a valid mesh"
        );
    }
}

/// The parallel skin stream must agree with the vertex buffer it is
/// index-aligned to. `SubMesh`'s own doc states the invariant; until now
/// nothing enforced it on the decode path, and both streams are uploaded as one
/// interleaved draw.
#[test]
fn a_decoded_inf_mesh_with_a_short_skin_stream_is_refused() {
    use inf_mesh::{MeshAsset, MeshVertex, SubMesh, VertexSkin};

    let mesh = |skin: Vec<VertexSkin>| {
        MeshAsset::new(
            vec![SubMesh {
                name: "s".into(),
                vertices: vec![MeshVertex::default(); 3],
                indices: vec![0, 1, 2],
                material_slot: None,
                skin,
            }],
            Vec::new(),
        )
    };

    // Empty is legal — that is a rigid submesh, and every pre-P11 payload.
    let bytes = inf_asset::encode(&mesh(Vec::new())).unwrap();
    inf_asset::decode::<MeshAsset>(&bytes).expect("an unskinned submesh must decode");
    // Aligned is legal.
    let bytes = inf_asset::encode(&mesh(vec![VertexSkin::default(); 3])).unwrap();
    inf_asset::decode::<MeshAsset>(&bytes).expect("an aligned skin stream must decode");
    // Short is not.
    let bytes = inf_asset::encode(&mesh(vec![VertexSkin::default(); 2])).unwrap();
    let e = inf_asset::decode::<MeshAsset>(&bytes)
        .expect_err("a skin stream shorter than its vertices decoded as valid");
    assert!(
        e.to_string().contains("2 elements against 3"),
        "the refusal must name both lengths: {e}"
    );
}
