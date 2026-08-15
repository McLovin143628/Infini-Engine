//! **The talus threshold is computed once, portably, for both erosion passes**
//! (Hardening Wave C, L6.F7).
//!
//! Thermal erosion asks one question per neighbour: is `diff / dist` steeper
//! than the material's angle of repose? The threshold is that angle's tangent,
//! and it is *not* computed where you would expect. `erosion.wgsl` never calls
//! `tan()` — it reads a `talus_tan` uniform — so the number the GPU compares
//! against is produced by Rust, in **this** crate, while the number the CPU
//! reference compares against is produced by Rust in `inf-terrain`.
//!
//! Those two lines therefore *are* the CPU/GPU parity envelope. Until Wave C
//! they were two copies of one expression, and `erosion_gpu`'s own parity gates
//! compare at `1e-3 × height_range` — three orders too loose to notice if one
//! copy drifted from the other. Now there is one function and the second crate
//! calls it, and this gate is what keeps that true.
//!
//! # And the expression itself was libm
//!
//! `f64::tan` is not bit-identical across targets (the P14 law). Eroded terrain
//! is **data**: `edit_region_with_maps` writes the result back into a
//! `TerrainData`, which is saved into an `.inf_terrain`, cooked into a pack and
//! shipped. A bake on the author's machine and a bake on a CI runner must not
//! produce two different worlds — and no gate in this repository compares two
//! machines, which is exactly why a source pin is the only thing that can see it.
//!
//! # Why source text
//!
//! Same technique and rationale as `grammar_span_mirror.rs` and
//! `projector_mirror.rs`: the property is about *which function the code is
//! allowed to call*, and no numeric assertion inside one process can see it.

use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
}

fn read(rel: &str) -> String {
    let path = workspace_root().join(rel);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
        .replace("\r\n", "\n")
}

/// The CPU reference pass, which owns [`inf_terrain::erosion::talus_tan`].
const CPU: &str = "crates/inf-terrain/src/erosion.rs";
/// The GPU host, which uploads the threshold as an `f32` uniform.
const GPU: &str = "editor/crates/inf-editor-core/src/erosion_gpu/mod.rs";
/// The shader, which must keep *reading* the uniform rather than deriving it.
const SHADER: &str = "editor/crates/inf-editor-core/src/erosion_gpu/erosion.wgsl";

/// A file's production half with comment lines blanked.
///
/// Comments are blanked because both Rust files now explain, at length, why
/// `tan` is banned — and a doc comment naming the thing it forbids is the
/// sentence a ban list is likeliest to be written next to (the P24.1 F1
/// finding). `#[cfg(test)]` and everything after it is cut for the usual
/// reason: a fixture may legitimately call `tan` to *check* the portable form,
/// and forcing it not to would make the fixture agree with the code under test
/// by construction.
fn production_code(rel: &str) -> String {
    let src = read(rel);
    let end = src.find("#[cfg(test)]").unwrap_or(src.len());
    src[..end]
        .lines()
        .map(|l| {
            let t = l.trim_start();
            if t.starts_with("//") {
                ""
            } else {
                l
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// **One list, not six** (round-2 finding R2.B).
///
/// This copy banned `f64::cos(` over a module whose parameters are `f32`, so
/// `f32::cos(` walked straight past it — the named hole that made the divergence
/// worth closing. `inf_math::libm_ban::ALL` is the one list and
/// `covers_both_spellings` is the completeness meta-arm.
const GATE: &str = "inf-editor-core/tests/erosion_talus_mirror.rs";

fn offenders(code: &str) -> Vec<&'static str> {
    let banned: Vec<&'static str> = inf_math::libm_ban::ALL.to_vec();
    inf_math::libm_ban::covers_both_spellings(GATE, &banned);
    banned.into_iter().filter(|b| code.contains(b)).collect()
}

/// Neither erosion pass derives its threshold with libm.
#[test]
fn the_talus_threshold_calls_no_libm_on_either_side() {
    for rel in [CPU, GPU] {
        let code = production_code(rel);
        let found = offenders(&code);
        assert!(
            found.is_empty(),
            "{rel} calls {found:?} — libm is not bit-identical across targets \
             (the P14 law), and eroded terrain is data that is saved, cooked and \
             shipped. Use `inf_terrain::erosion::talus_tan`, which goes through \
             `inf_math`'s portable pair."
        );
    }
}

/// **One door, and the second crate really walks through it.**
///
/// The CPU pass owns the function; the GPU host calls it and narrows once at the
/// wire. Anything else — a second `to_radians().tan()`, a hand-inlined
/// polynomial, a constant — reopens the envelope this gate exists to close.
#[test]
fn both_erosion_passes_read_one_threshold() {
    let cpu = production_code(CPU);
    assert!(
        cpu.contains("pub fn talus_tan(deg: f32) -> f64"),
        "`inf_terrain::erosion::talus_tan` is gone or changed shape; the GPU host \
         calls it by name"
    );
    assert!(
        cpu.contains("let talus_tan = talus_tan(p.thermal_talus_deg);"),
        "the CPU thermal pass no longer computes its threshold through its own \
         function"
    );

    let gpu = production_code(GPU);
    assert!(
        gpu.contains("inf_terrain::erosion::talus_tan(params.thermal_talus_deg) as f32"),
        "the GPU host no longer uploads the CPU reference's own threshold — these \
         two lines ARE the CPU/GPU parity envelope, and `erosion_gpu`'s parity \
         arms compare at 1e-3 of the height range, far too loose to see them \
         disagree"
    );

    // **The shader must keep reading the uniform.** If it ever derived the
    // threshold itself, the two sides would be a Rust polynomial and a GPU
    // vendor's `tan` — a divergence with no fix short of removing it again.
    let shader = read(SHADER);
    let shader_code: String = shader
        .lines()
        .map(|l| {
            if l.trim_start().starts_with("//") {
                ""
            } else {
                l
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !shader_code.contains("tan("),
        "erosion.wgsl now calls `tan()`; the talus threshold must arrive as the \
         `P.talus_tan` uniform the host computes"
    );
    assert!(
        shader_code.contains("P.talus_tan"),
        "erosion.wgsl no longer reads the threshold uniform at all — this gate is \
         pointing at a shader that has stopped doing thermal erosion"
    );
}

/// **NOT VACUOUS**: the files exist, they contain code, and the ban rejects the
/// call it was written for.
#[test]
fn the_talus_pin_is_looking_at_real_code() {
    for rel in [CPU, GPU] {
        let code = production_code(rel);
        assert!(
            code.lines().filter(|l| !l.trim().is_empty()).count() > 200,
            "{rel} reduced to almost nothing after the split — the ban is \
             scanning an empty string"
        );
    }
    // Built to falsify (the P22 law): the same predicate over a poisoned copy
    // must reject it. The poison is the exact defect this gate was written for.
    let poisoned = format!(
        "{}\n    talus_tan: (params.thermal_talus_deg as f64).to_radians().tan() as f32,",
        production_code(GPU)
    );
    assert!(
        !offenders(&poisoned).is_empty(),
        "the ban cannot see a `tan` even when the second copy is put back in \
         front of it"
    );
}
