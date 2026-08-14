//! The one atomic file-write door.
//!
//! Every document this engine writes — a `.inf_lvl`, a sidecar, a project
//! `inf.toml`, a dock layout, the crash-recovery file, a user's own `.rs` under
//! Ctrl+S — goes through [`write_atomically`]. A plain `std::fs::write`
//! *truncates first and fills afterwards*, so a power loss, an OOM kill or a
//! full disk mid-call leaves the file that was there **destroyed** and the file
//! that was coming **absent**. For a build artifact that is a rebuild; for a
//! document it is the user's work.
//!
//! The mechanism is the one [`crate::pack::PackWriter::write_to_file`] has used
//! since P16.1: write the whole payload to a *sibling* temp file (same
//! directory, so the rename cannot cross a filesystem and is therefore atomic),
//! then `rename` it over the destination. The destination is only ever the whole
//! old file or the whole new one — never a prefix of either — and a reader that
//! already has the old file open keeps reading the old bytes rather than
//! watching them change underneath it.
//!
//! # The temp name carries a pid **and** a counter
//!
//! Spike C's shadow-copy race is the law here: two writers (two processes, or
//! two threads of one) that pick the same temp name will interleave a partial
//! payload into each other's rename. The pid separates processes; the
//! process-wide counter separates threads. A name with only one of the two is a
//! race with a longer fuse, not a safe name.
//!
//! # Durability is deliberately not promised
//!
//! There is no `fsync` before the rename. The guarantee is *atomicity* — no
//! reader ever sees a torn file — not *durability* across a power cut, which
//! would cost a full flush on every keystroke-driven autosave. A caller that
//! genuinely needs the stronger promise must say so at its own door.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// A unique temp path beside `path`: same directory (so the rename that follows
/// stays on one filesystem), name suffixed with `.<pid>.<counter>.tmp`.
///
/// Both halves are load-bearing — see the module docs. The counter is
/// process-wide rather than per-call-site so that two different doors writing to
/// one directory in the same millisecond cannot collide either.
pub fn temp_sibling(path: &Path) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let unique = NEXT.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".{pid}.{unique}.tmp"));
    path.with_file_name(name)
}

/// Write `bytes` to `path` **atomically**: parent directories created, payload
/// staged in a [`temp_sibling`], then renamed over the destination.
///
/// On any failure the temp file is removed — a failed write leaves the directory
/// exactly as it found it, with neither litter nor a damaged destination — and
/// the original error is returned unchanged, so a caller's message ("could not
/// write X … nothing on disk changed") is true rather than aspirational.
pub fn write_atomically(path: &Path, bytes: impl AsRef<[u8]>) -> io::Result<()> {
    // `Path::new("a.txt").parent()` is `Some("")`, and `create_dir_all("")`
    // fails on Windows — a relative single-segment destination is legal and must
    // not be turned into an error by the convenience half of this function.
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let tmp = temp_sibling(path);
    let staged = std::fs::write(&tmp, bytes.as_ref()).and_then(|()| std::fs::rename(&tmp, path));
    if let Err(err) = staged {
        let _ = std::fs::remove_file(&tmp);
        return Err(err);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every temp name is unique within a process (the counter half of the Spike
    /// C law) and carries the pid (the process half).
    #[test]
    fn temp_names_carry_a_pid_and_never_repeat() {
        let dir = Path::new("/content/Levels");
        let target = dir.join("main.inf_lvl");
        let pid = std::process::id().to_string();
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..64 {
            let tmp = temp_sibling(&target);
            assert_eq!(tmp.parent(), target.parent(), "temp must be a SIBLING");
            let name = tmp.file_name().unwrap().to_string_lossy().into_owned();
            assert!(name.ends_with(".tmp"), "{name}");
            assert!(
                name.contains(&format!(".{pid}.")),
                "temp name must carry the pid: {name}"
            );
            assert!(seen.insert(name.clone()), "duplicate temp name {name}");
        }
    }

    /// The whole point, on the exact shape the engine writes: a reader that
    /// already holds the destination open never observes the rewrite.
    ///
    /// This is the arm that dies if `write_atomically` is downgraded to a plain
    /// `std::fs::write` — the held handle would then read the *new* bytes (or a
    /// truncated prefix of them), which is precisely the torn read a document
    /// must never expose.
    #[test]
    fn a_reader_holding_the_old_file_never_observes_the_rewrite() {
        use std::io::Read;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("main.inf_lvl");
        let old = vec![b'A'; 64 * 1024];
        let new = vec![b'B'; 64 * 1024];
        write_atomically(&path, &old).unwrap();

        let mut live = std::fs::File::open(&path).unwrap();
        write_atomically(&path, &new).unwrap();

        let mut seen = Vec::new();
        live.read_to_end(&mut seen).unwrap();
        assert_eq!(seen, old, "a live reader must not observe the rewrite");
        assert_eq!(
            std::fs::read(&path).unwrap(),
            new,
            "the file is the new one"
        );
        assert!(no_temp_litter(dir.path()), "left temp files behind");
    }

    /// A write that cannot land destroys nothing and leaves nothing behind.
    ///
    /// The destination is a **directory** with a file in it: `create_dir_all`
    /// succeeds, the temp write succeeds, and the rename is the step that fails
    /// (`ENOTEMPTY`/`EISDIR` on Unix, access denied on Windows) — so the arm
    /// exercises the cleanup path rather than an early return, on every
    /// platform.
    #[test]
    fn a_failed_write_destroys_nothing_and_leaves_no_litter() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("occupied");
        std::fs::create_dir_all(&dest).unwrap();
        std::fs::write(dest.join("keep.txt"), b"survivor").unwrap();

        let err = write_atomically(&dest, b"new bytes").unwrap_err();
        assert!(dest.is_dir(), "the destination survived: {err}");
        assert_eq!(std::fs::read(dest.join("keep.txt")).unwrap(), b"survivor");
        assert!(no_temp_litter(dir.path()), "left temp files behind");
    }

    /// A read-only destination refuses the rename and the old bytes survive
    /// whole — the Windows spelling of "the write failed midway".
    ///
    /// Windows-only on purpose: on Unix the permission that governs `rename` is
    /// the *directory's*, not the file's, so the same test there would rename
    /// straight over a read-only file and assert nothing.
    #[cfg(windows)]
    #[test]
    fn a_read_only_destination_keeps_its_old_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("locked.inf_lvl");
        write_atomically(&path, b"original").unwrap();
        let mut perms = std::fs::metadata(&path).unwrap().permissions();
        perms.set_readonly(true);
        std::fs::set_permissions(&path, perms).unwrap();

        assert!(
            write_atomically(&path, b"replacement").is_err(),
            "a read-only destination must refuse the write"
        );
        assert_eq!(std::fs::read(&path).unwrap(), b"original");
        assert!(no_temp_litter(dir.path()), "left temp files behind");

        // Clear the bit again so the `TempDir` can actually delete itself —
        // Windows refuses to unlink a read-only file, and a leaked one would
        // sit in `%TEMP%` for ever.
        //
        // `clippy::permissions_set_readonly_false` fires because on **Unix**
        // this clears every write bit to world-writable. This whole test is
        // `#[cfg(windows)]`, where the flag is the single FILE_ATTRIBUTE_READONLY
        // bit and clearing it restores exactly the state four lines of this test
        // created, on a file inside a temp directory that is about to be removed.
        // The narrow allow is the accurate statement; a `PermissionsExt` import
        // would not compile on the only platform this test exists for.
        #[allow(clippy::permissions_set_readonly_false)]
        {
            let mut perms = std::fs::metadata(&path).unwrap().permissions();
            perms.set_readonly(false);
            std::fs::set_permissions(&path, perms).unwrap();
        }
    }

    /// Renaming **over an existing file** is the mechanism's whole premise, and
    /// it is exactly the premise `layouts::save` doubted (C4-23) when it deleted
    /// the destination first. It holds on Windows.
    #[test]
    fn rename_over_an_existing_file_replaces_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("preset.json");
        std::fs::write(&path, b"old").unwrap();
        let tmp = temp_sibling(&path);
        std::fs::write(&tmp, b"new").unwrap();
        std::fs::rename(&tmp, &path).expect("rename over an existing file must succeed");
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
        assert!(!tmp.exists());
    }

    fn no_temp_litter(dir: &Path) -> bool {
        std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .all(|e| !e.file_name().to_string_lossy().ends_with(".tmp"))
    }
}
