//! **The viewport command-pump MIRROR gate** (P21.2).
//!
//! `inf-viewport` has two platform modules — `win32.rs` and `macos.rs` — and each
//! carries its own copy of the same three-part structure: a `Cmd` enum, a
//! `ViewportHandle` method per variant, and a dispatch arm per variant in the
//! thread's pump. Ring 2 calls the handle methods **unconditionally**, so a
//! variant added on one side only is not a missing feature: it is a build that
//! does not compile on the other OS.
//!
//! That has happened. P20.4 (`c4bd663`) added `set_water` / `set_water_hints` to
//! win32 alone; macOS CI failed on two missing methods, and the fix (`adced6b`)
//! was a whole follow-up commit. This gate turns that into a local test failure
//! on any machine, in under a millisecond, instead of a red CI leg twenty minutes
//! later — and it does so **on all three OSes**, because it reads source text
//! rather than compiling either module (`inf_viewport::host` is
//! `#[cfg(any(windows, target_os = "macos"))]`, so a test inside that crate is
//! invisible to exactly the leg most likely to run first).
//!
//! What is pinned is the *set*, not the order: the two files list their variants
//! in the same relative positions by convention, but a diff in ordering is a style
//! question while a diff in membership is a broken build.
//!
//! Deliberately NOT pinned: the pump bodies. macOS input is not wired yet, so its
//! handle methods carry a standing "sets host state only" caveat and several of
//! its arms do less than win32's. The gate is that every command **exists** on
//! both sides and reaches the host.

use std::path::{Path, PathBuf};

fn read(rel: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join(rel);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", PathBuf::from(&path).display()))
        .replace("\r\n", "\n")
}

const WIN32: &str = "editor/crates/inf-viewport/src/win32.rs";
const MACOS: &str = "editor/crates/inf-viewport/src/macos.rs";
/// The no-op stub Ring 2 compiles against on platforms with no embedding backend
/// (Linux). A handle method missing here is a Linux build failure.
const LIB: &str = "editor/crates/inf-viewport/src/lib.rs";

/// Variant names of the `enum Cmd { … }` in `source`.
///
/// A variant is a line at the enum's indentation that starts with an upper-case
/// identifier — doc comments, attributes and blank lines are skipped, and the
/// payload (if any) is dropped.
fn cmd_variants(source: &str) -> Vec<String> {
    let start = source
        .find("enum Cmd {")
        .expect("no `enum Cmd {` — was the pump renamed?");
    let body = &source[start..];
    let end = body.find("\n}\n").expect("`enum Cmd` does not terminate");
    let mut out = Vec::new();
    for line in body[..end].lines().skip(1) {
        let t = line.trim();
        if t.is_empty() || t.starts_with("//") || t.starts_with('#') {
            continue;
        }
        let name: String = t
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if name.chars().next().is_some_and(|c| c.is_uppercase()) {
            out.push(name);
        }
    }
    assert!(out.len() > 10, "only {} variants parsed", out.len());
    out.sort();
    out
}

/// `pub fn` names declared in `source` — every handle method, in sorted order.
fn pub_fn_names(source: &str) -> Vec<String> {
    let mut out: Vec<String> = source
        .match_indices("pub fn ")
        .map(|(i, _)| {
            source[i + "pub fn ".len()..]
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect()
        })
        .collect();
    out.sort();
    out.dedup();
    out
}

/// Commands that legitimately exist on **one** platform only, each with the
/// reason. Their `ViewportHandle` methods still exist on both sides — that is
/// what Ring 2 calls — but macOS answers them without a pump round trip.
///
/// * `EmbedForeign` / `ReleaseForeign` — embedded PIE (P9.4) is Win32
///   `SetParent` reparenting of the player's HWND into the editor's window
///   hierarchy. macOS has no equivalent that would not be a different design
///   (a child `NSWindow`, or an out-of-process layer), so its handle methods
///   are documented no-ops and there is nothing for a pump arm to do. A
///   variant there would be a command that is queued and ignored, which is
///   strictly worse than not having one.
const PLATFORM_ONLY: [&str; 2] = ["EmbedForeign", "ReleaseForeign"];

/// **The membership gate.** Both platform pumps must carry the same `Cmd` set,
/// modulo the documented [`PLATFORM_ONLY`] exemptions.
#[test]
fn both_platform_pumps_carry_the_same_commands() {
    let win: Vec<String> = cmd_variants(&read(WIN32))
        .into_iter()
        .filter(|v| !PLATFORM_ONLY.contains(&v.as_str()))
        .collect();
    let mac = cmd_variants(&read(MACOS));
    let only_win: Vec<&String> = win.iter().filter(|v| !mac.contains(v)).collect();
    let only_mac: Vec<&String> = mac.iter().filter(|v| !win.contains(v)).collect();
    assert!(
        only_win.is_empty() && only_mac.is_empty(),
        "the two viewport command pumps have drifted.\n  win32 only: {only_win:?}\n  macOS only: \
         {only_mac:?}\nRing 2 calls `ViewportHandle` methods unconditionally, so a command on one \
         side only is a build failure on the other OS — not a missing feature. Add the variant, \
         the handle method and the pump arm to both files (see `adced6b`, which is what this gate \
         exists to make unnecessary), or — if the difference is deliberate — add it to \
         `PLATFORM_ONLY` with the reason written down."
    );
}

/// A guard on the exemption list: every `PLATFORM_ONLY` name must actually be a
/// win32 command, so a stale entry cannot quietly hide a real drift under the
/// name of a variant that no longer exists.
#[test]
fn the_platform_only_exemptions_are_all_real() {
    let win = cmd_variants(&read(WIN32));
    for name in PLATFORM_ONLY {
        assert!(
            win.iter().any(|v| v == name),
            "`{name}` is exempted from the pump mirror but is not a win32 command — \
             the exemption is stale and would mask a drift with that name"
        );
    }
}

/// **The dispatch gate.** Every variant must actually be handled — a variant that
/// reaches the pump and matches nothing is a command that silently does nothing.
#[test]
fn every_command_is_dispatched_on_both_platforms() {
    for (label, path) in [("win32", WIN32), ("macOS", MACOS)] {
        let src = read(path);
        for v in cmd_variants(&src) {
            assert!(
                src.contains(&format!("Cmd::{v}")),
                "the {label} pump never dispatches `Cmd::{v}` — the command would be \
                 accepted, queued and dropped"
            );
        }
    }
}

/// **The Linux-stub gate.** Ring 2 compiles against `lib.rs`'s no-op
/// `ViewportHandle` on platforms with no embedding backend, so every method the
/// two real handles expose has to exist there too.
///
/// The stub is where a forgotten method fails *last* and most confusingly: the
/// editor builds and runs on the developer's machine and the Linux CI leg reports
/// a missing method in a file nobody touched.
#[test]
fn the_linux_stub_carries_every_handle_method() {
    let win = pub_fn_names(&read(WIN32));
    let stub = pub_fn_names(&read(LIB));
    // `spawn` is a free function on both real backends and on the stub; it is a
    // constructor rather than a handle method, so it rides along either way.
    let missing: Vec<&String> = win
        .iter()
        .filter(|f| !stub.contains(f) && f.as_str() != "spawn")
        .collect();
    assert!(
        missing.is_empty(),
        "the Linux no-op `ViewportHandle` in lib.rs is missing {missing:?} — Ring 2 \
         calls these unconditionally, so that leg will not build"
    );
}

/// A guard on the guards: the P21.2 voxel commands really are the pair this batch
/// added, on every one of the three surfaces.
///
/// Without this, all three tests above would still pass if the voxel tool had
/// never been wired at all — they compare the files to each other, and two files
/// that both lack a command agree perfectly.
#[test]
fn the_voxel_commands_reach_every_surface() {
    for (label, path) in [("win32", WIN32), ("macOS", MACOS)] {
        let src = read(path);
        for fragment in [
            "SetVoxel(VoxelSettings)",
            "ReloadVoxelStores",
            "pub fn set_voxel(",
            "pub fn reload_voxel_stores(",
            "host.set_voxel(",
            "host.reload_voxel_stores()",
        ] {
            assert!(
                src.contains(fragment),
                "the {label} pump is missing `{fragment}` (P21.2)"
            );
        }
    }
    let stub = read(LIB);
    for fragment in ["pub fn set_voxel(", "pub fn reload_voxel_stores("] {
        assert!(
            stub.contains(fragment),
            "the Linux stub is missing `{fragment}`"
        );
    }
}
