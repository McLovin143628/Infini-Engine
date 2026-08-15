//! **The import door** — the one place an external mesh document is checked
//! before any of it becomes engine data.
//!
//! # Why one door, and why here
//!
//! `.gltf`, `.glb` and `.obj` are the only formats in this engine that carry
//! *foreign* numbers. Every other producer of geometry — the DCC kernel, the
//! PCG grammar, the voxel mesher, photogrammetry — computes its vertices from
//! values the engine already owns, so its output is wrong only if the engine is
//! wrong. An imported file is not like that: it is bytes somebody else wrote,
//! and this crate hands them to three things that cannot defend themselves.
//!
//! * **`meshopt`.** [`crate::optimize()`] passes the index buffer to a C library
//!   through raw FFI. `generate_vertex_remap` sizes its remap table from
//!   `vertices.len()` and the C side writes `remap[index]` with its `assert`
//!   compiled out under `-DNDEBUG` — so one index past the end of the vertex
//!   buffer is an out-of-bounds heap **write**, not a panic.
//! * **The GPU.** [`crate::VertexSkin`] and [`crate::MeshVertex`] are `Pod` and
//!   are uploaded verbatim.
//! * **Every committed byte downstream.** A NaN that gets past here is written
//!   into `.inf_mesh` / `.inf_skel` / `.inf_anim` / `.inf_mat`, and from there
//!   into the meshlet DAG's cull bounds, the DCC kernel's weld keys, and
//!   `pose_state_bytes` — the trace the PIE-==-shipping parity gate compares.
//!   `f32::min`/`max` ignore NaN, so the bounding box computed from a poisoned
//!   vertex buffer looks perfectly healthy; nothing downstream is looking.
//!
//! Rust's own `str::parse::<f32>` accepts `"nan"`, `"inf"` and `"infinity"`, so
//! `v nan nan nan` is a *syntactically valid* OBJ vertex line, and a glTF
//! accessor is raw IEEE-754 out of a binary buffer with no parser between it and
//! us at all. Neither format has a rule against it. This module is that rule.
//!
//! # Shape of the checks
//!
//! Three questions, asked with the answer NaN gives:
//!
//! 1. **Finiteness** — [`reject_non_finite`]. Written as `is_finite()`, never as
//!    a comparison, because the house's standing NaN-blind-guard class is
//!    exactly a comparison that a NaN answers `false` and therefore passes.
//! 2. **Index bounds** — [`reject_out_of_range`] / [`reject_joint_indices`] —
//!    and index **arity**, [`reject_partial_triangle`]: every `meshopt` entry
//!    asserts `index_count % 3 == 0`, and that assert is compiled out, so a
//!    partial triangle is an out-of-bounds heap write rather than a diagnostic.
//! 3. **Stream-length agreement** — [`reject_length_mismatch`]. A glTF sampler
//!    whose output count disagrees with its input count, or a `TEXCOORD_0`
//!    shorter than `POSITION`, is a file whose streams do not describe the same
//!    vertices.
//!
//! Every refusal names the file's problem in the file's own vocabulary (which
//! attribute, which element), because the person reading it is holding an
//! exporter, not this source tree.

use crate::error::MeshError;

/// Anything an import stream can hold whose every float component must be a real
/// number.
///
/// Implemented for `f32` and, recursively, for arrays — so one bound covers a
/// `VEC2` uv stream, a `VEC4` weight stream and a `MAT4` inverse-bind stream
/// without a check per shape.
pub trait AllFinite {
    /// Whether every float this value contains is finite (neither NaN nor ±∞).
    fn all_finite(&self) -> bool;
}

impl AllFinite for f32 {
    fn all_finite(&self) -> bool {
        self.is_finite()
    }
}

impl<T: AllFinite, const N: usize> AllFinite for [T; N] {
    fn all_finite(&self) -> bool {
        self.iter().all(AllFinite::all_finite)
    }
}

/// Refuse a stream carrying a NaN or an infinity, naming the first offender.
///
/// `what` is the source file's name for the stream (`"POSITION"`, `"v"`,
/// `"inverseBindMatrices"`), not ours.
pub fn reject_non_finite<T: AllFinite>(values: &[T], what: &str) -> Result<(), MeshError> {
    match values.iter().position(|v| !v.all_finite()) {
        Some(i) => Err(MeshError::Malformed(format!(
            "{what}: element {i} is not a finite number (NaN or infinity)"
        ))),
        None => Ok(()),
    }
}

/// Refuse an index buffer that addresses outside its vertex buffer.
///
/// Also refuses a non-empty index buffer over an **empty** vertex buffer, which
/// is the shape a primitive with a zero-count `POSITION` accessor and a
/// populated index accessor produces — it would otherwise be persisted as a
/// perfectly parseable `.inf_mesh` whose every index dangles.
pub fn reject_out_of_range(
    indices: &[u32],
    vertex_count: usize,
    what: &str,
) -> Result<(), MeshError> {
    if indices.is_empty() {
        return Ok(());
    }
    if vertex_count == 0 {
        return Err(MeshError::Malformed(format!(
            "{what}: {} indices over an empty vertex buffer",
            indices.len()
        )));
    }
    match indices.iter().position(|&i| i as usize >= vertex_count) {
        Some(at) => Err(MeshError::Malformed(format!(
            "{what}: index {} at position {at} addresses outside a {vertex_count}-vertex buffer",
            indices[at]
        ))),
        None => Ok(()),
    }
}

/// Refuse an index buffer that is not a whole number of triangles.
///
/// # The compiled-out assert, and the underflow behind it
///
/// Every `meshopt` entry point this workspace calls opens with
/// `assert(index_count % 3 == 0)`, and the vendored build compiles its asserts
/// out — so the library does not check this, it *assumes* it. Three separate
/// consequences follow, all verified in `meshopt-0.4.1/vendor/src`:
///
/// * **An out-of-bounds heap write.** `meshopt_optimizeVertexCacheTable`
///   computes `face_count = index_count / 3`, then enters its emit loop over a
///   second compiled-out `assert(output_triangle < face_count)` and reads
///   `indices[current_triangle * 3 + 2]`. With two indices that is one `u32`
///   past the end of the input **and** one past the end of the destination the
///   Rust wrapper allocated at `indices.len()`. Two indices is what a glTF
///   primitive with a count-2 index accessor produces.
/// * **An underflow.** `meshopt::build_meshlets` finishes with
///   `meshlets[count - 1]`; a floored `index_count / 3` of zero returns
///   `count == 0`, and `0 - 1` is `usize::MAX`.
/// * **Silent geometry loss.** The entry points that floor drop the trailing
///   one or two indices, while `optimize_vertex_cache`'s wrapper still returns
///   a buffer of the original length — so the tail comes back as a fabricated
///   `[0, 0, …]` triangle that was in nobody's file.
///
/// A partial triangle is meaningless in every format this engine reads (glTF's
/// `TRIANGLES` mode and `.obj`'s `f` both describe whole faces), so there is no
/// legitimate producer to accommodate. `indices.is_empty()` is not a partial
/// triangle — an unindexed submesh is a real state.
pub fn reject_partial_triangle(index_count: usize, what: &str) -> Result<(), MeshError> {
    if index_count.is_multiple_of(3) {
        Ok(())
    } else {
        Err(MeshError::Malformed(format!(
            "{what}: {index_count} indices is not a whole number of triangles ({} left over)",
            index_count % 3
        )))
    }
}

/// Refuse skinning influences that name a joint the skeleton does not have.
///
/// Nothing panics on these — the GPU path is naga-bounds-checked and the CPU
/// path clamps — so the symptom is a silently wrong deformation, which is worse
/// than a refusal precisely because it looks like the author's fault.
pub fn reject_joint_indices(
    joints: &[[u16; 4]],
    joint_count: usize,
    what: &str,
) -> Result<(), MeshError> {
    if joint_count == 0 {
        // No skeleton was resolved for this primitive; the influences are
        // dropped rather than bound, so there is nothing to bound them against.
        return Ok(());
    }
    for (i, j) in joints.iter().enumerate() {
        if let Some(&bad) = j.iter().find(|&&b| b as usize >= joint_count) {
            return Err(MeshError::Malformed(format!(
                "{what}: vertex {i} is weighted to joint {bad}, but the skin has {joint_count}"
            )));
        }
    }
    Ok(())
}

/// Refuse two streams that are supposed to describe the same elements and do
/// not.
pub fn reject_length_mismatch(
    len: usize,
    expected: usize,
    what: &str,
    against: &str,
) -> Result<(), MeshError> {
    if len == expected {
        Ok(())
    } else {
        Err(MeshError::Malformed(format!(
            "{what}: {len} elements against {expected} in {against}"
        )))
    }
}

/// Refuse a keyframe time list that is not strictly increasing.
///
/// glTF requires it, and every consumer in `inf-anim` assumes it:
/// `clip::locate` binary-searches with `partition_point`, so an unsorted list
/// silently returns the wrong bracket and a NaN makes **both** of its range
/// guards false — which used to underflow `hi - 1` and panic in the shipped
/// player. Equal adjacent times are refused too: they are a zero-length span,
/// and a span is a divisor.
pub fn reject_non_increasing(times: &[f32], what: &str) -> Result<(), MeshError> {
    reject_non_finite(times, what)?;
    for i in 1..times.len() {
        // `reject_non_finite` above has already established that every time is a
        // real number, so this ordinary comparison is exact — a negated one
        // would be neither clearer nor needed.
        if times[i] <= times[i - 1] {
            return Err(MeshError::Malformed(format!(
                "{what}: keyframe time {} at {i} does not increase past {} at {}",
                times[i],
                times[i - 1],
                i - 1
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finiteness_is_asked_as_a_predicate_not_a_comparison() {
        // The whole point of the class: every ordering comparison a NaN takes
        // part in is false, so a guard shaped like one lets it through.
        assert!(reject_non_finite(&[[0.0f32, 1.0, 2.0]], "POSITION").is_ok());
        for poison in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let v = [[0.0f32, poison, 0.0]];
            let e = reject_non_finite(&v, "POSITION").unwrap_err();
            assert!(
                e.to_string().contains("POSITION") && e.to_string().contains("element 0"),
                "refusal did not name the stream and the element: {e}"
            );
        }
        // The recursive array impl reaches into a MAT4 without a per-shape check.
        let mut m = [[0.0f32; 4]; 4];
        m[3][2] = f32::NAN;
        assert!(reject_non_finite(&[m], "inverseBindMatrices").is_err());
    }

    #[test]
    fn an_index_past_the_vertex_buffer_is_refused_before_meshopt_sees_it() {
        assert!(reject_out_of_range(&[0, 1, 2], 3, "indices").is_ok());
        let e = reject_out_of_range(&[0, 1, 3], 3, "indices").unwrap_err();
        assert!(e.to_string().contains("index 3"), "{e}");
        // The zero-vertex primitive: indices over nothing.
        assert!(reject_out_of_range(&[0, 1, 2], 0, "indices").is_err());
        // An empty index buffer over an empty vertex buffer is not a defect.
        assert!(reject_out_of_range(&[], 0, "indices").is_ok());
    }

    #[test]
    fn an_index_count_that_is_not_whole_triangles_is_refused() {
        for whole in [0usize, 3, 6, 300] {
            assert!(reject_partial_triangle(whole, "indices").is_ok(), "{whole}");
        }
        for partial in [1usize, 2, 4, 5, 301] {
            let e = reject_partial_triangle(partial, "indices").unwrap_err();
            let msg = e.to_string();
            assert!(
                msg.contains("whole number of triangles") && msg.contains(&partial.to_string()),
                "the refusal must name the count: {msg}"
            );
        }
    }

    #[test]
    fn joint_indices_are_bounded_against_the_skin_that_resolved() {
        assert!(reject_joint_indices(&[[0, 1, 2, 0]], 3, "JOINTS_0").is_ok());
        assert!(reject_joint_indices(&[[0, 1, 3, 0]], 3, "JOINTS_0").is_err());
        // Unresolved skin: nothing to bound against, and the influences are
        // dropped rather than bound.
        assert!(reject_joint_indices(&[[9, 9, 9, 9]], 0, "JOINTS_0").is_ok());
    }

    #[test]
    fn keyframe_times_must_strictly_increase() {
        assert!(reject_non_increasing(&[0.0, 0.5, 1.0], "input").is_ok());
        assert!(reject_non_increasing(&[0.0, 0.5, 0.5], "input").is_err());
        assert!(reject_non_increasing(&[0.0, 1.0, 0.5], "input").is_err());
        assert!(reject_non_increasing(&[0.0, f32::NAN], "input").is_err());
        // A single key and an empty list are both trivially increasing.
        assert!(reject_non_increasing(&[7.0], "input").is_ok());
        assert!(reject_non_increasing(&[], "input").is_ok());
    }

    #[test]
    fn stream_lengths_are_compared_against_the_stream_they_index() {
        assert!(reject_length_mismatch(3, 3, "TEXCOORD_0", "POSITION").is_ok());
        let e = reject_length_mismatch(2, 3, "TEXCOORD_0", "POSITION").unwrap_err();
        assert!(e.to_string().contains("2 elements against 3"), "{e}");
    }
}
