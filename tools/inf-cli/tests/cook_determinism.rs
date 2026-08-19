//! **Two independent cooks, in two independent PROCESSES** (P29.7, closing the
//! P29.6 audit's subprocess-pool gap).
//!
//! # Why this arm is here and not in `phase29_gate`
//!
//! `phase29_gate`'s arm (b) cooks the showcase course twice and replays both
//! packs, and the P29.6 audit named its bound: the two cooks are two temp
//! directories **in one process**, so every `OnceLock` pool, every in-memory
//! import cache and every lazily-initialized global is shared between them. Two
//! cooks that agree because they are the same cook prove nothing about a cook
//! run tomorrow on a clean machine. This repository's own law, met at P20.2 and
//! again at P22.4: *a pool gate needs subprocesses*.
//!
//! Closing it needs a **binary that can cook**, and the shipped player is
//! deliberately not one — `inf-packager` is a dev-dependency of `inf-player`
//! with a comment saying so, because the ring rule (a player never links the
//! cook pipeline) outranks the convenience of putting every arm of a phase gate
//! in one file. The binary that can cook is `inf`, its tests already spawn it
//! through `CARGO_BIN_EXE_inf`, and so the arm lives beside the binary rather
//! than beside its sibling claims. `phase29_gate`'s arm (b) names this file.
//!
//! # What it asserts
//!
//! The **bytes**, which is a stronger claim than the trace comparison it backs:
//! if two independently-cooked packs are byte-identical then any two replays off
//! them are identical for free, and a pipeline that made a different choice — an
//! id, an ordering, a derived byte, a hash-map walk — is caught at the pack
//! rather than at the far end of five thousand simulated steps.

use std::path::{Path, PathBuf};
use std::process::Command;

fn inf() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_inf"))
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// Scaffold a project whose `Content` is the committed showcase course — the
/// same files `phase29_gate` cooks, copied rather than generated so this arm
/// reads what is on disk.
fn scaffold(root: &Path) -> usize {
    std::fs::create_dir_all(root.join("Content")).unwrap();
    std::fs::write(
        root.join("inf.toml"),
        "schema_version = 1\nname = \"Phase29Locomotion\"\nengine_version = \"0.1.0\"\ntemplate = \"blank-3d\"\n",
    )
    .unwrap();
    let sample = workspace_root().join("samples/phase29-locomotion");
    let mut copied = 0;
    for entry in std::fs::read_dir(&sample).expect("the committed sample is on disk") {
        let entry = entry.unwrap();
        if !entry.file_type().unwrap().is_file() {
            continue;
        }
        std::fs::copy(entry.path(), root.join("Content").join(entry.file_name())).unwrap();
        copied += 1;
    }
    copied
}

/// Cook `proj` in a **subprocess** and answer its pack and manifest bytes.
fn cook_in_a_subprocess(proj: &Path) -> (Vec<u8>, Vec<u8>) {
    let out = Command::new(inf())
        .args(["cook", "--project"])
        .arg(proj)
        .output()
        .expect("inf cook runs");
    assert!(
        out.status.success(),
        "cook exit: {:?}\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let build = proj.join("Build");
    (
        std::fs::read(build.join("content.inf_pack")).expect("the pack was written"),
        std::fs::read(build.join("manifest.toml")).expect("the manifest was written"),
    )
}

#[test]
fn two_cooks_in_two_processes_produce_the_same_pack_byte_for_byte() {
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a");
    let b = dir.path().join("b");
    let copied = scaffold(&a);
    assert_eq!(scaffold(&b), copied);
    // The fixture must be the real course, not an empty folder that two cooks
    // agree about perfectly: nineteen committed files, and a pack big enough to
    // hold a rig, three derived clips, a machine, a controller and a level.
    assert!(
        copied >= 19,
        "only {copied} files were copied out of the committed sample"
    );

    let (pack_a, manifest_a) = cook_in_a_subprocess(&a);
    let (pack_b, manifest_b) = cook_in_a_subprocess(&b);
    assert!(
        pack_a.len() > 4096,
        "the pack is {} bytes — that is not this course",
        pack_a.len()
    );
    assert_eq!(
        pack_a.len(),
        pack_b.len(),
        "two independent cooks produced packs of different sizes"
    );
    assert!(
        pack_a == pack_b,
        "two independent cooks produced different pack BYTES — the pipeline made \
         a choice that depends on something outside its inputs (an id, an \
         ordering, a hash walk, a clock)"
    );
    // The manifest carries the root level's guid and the pack name; a cook that
    // minted a fresh id per run would pass the pack compare only if the id never
    // reached the payload, so both are checked.
    assert_eq!(
        String::from_utf8_lossy(&manifest_a),
        String::from_utf8_lossy(&manifest_b),
        "two independent cooks produced different manifests"
    );
}
