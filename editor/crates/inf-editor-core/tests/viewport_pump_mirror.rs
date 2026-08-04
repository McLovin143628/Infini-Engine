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
//! # What this gate does NOT cover — read this before trusting it
//!
//! **It compares source TEXT, not behaviour.** Every assertion below is either a
//! set comparison of parsed names or a `contains` on a fragment. Specifically:
//!
//! * **Pump bodies are not compared.** macOS input is not wired yet, so its
//!   handle methods carry a standing "sets host state only" caveat and several of
//!   its arms deliberately do less than win32's. A macOS arm that dispatches
//!   `Cmd::SetVoxel` into a call that does the wrong thing passes every test here,
//!   and so does a win32 arm that stops updating anything. The gate is that every
//!   command **exists** on both sides and is named in a `Cmd::…` pattern.
//! * **Interaction is not mirrored at all**, because macOS has no interaction
//!   block (no input). [`the_voxel_commands_reach_every_surface`] therefore checks
//!   win32's carve calls one-sidedly and only that they are *present* — it cannot
//!   tell whether they run in the right order, under the right gate, or at all at
//!   runtime. That is the platform pass's job.
//! * **A fragment list is only as good as its last update.** The P21.2 audit
//!   proved the point: `pub fn voxel_volumes` — the one new handle surface this
//!   batch added — was missing from the list, so deleting it broke the build while
//!   every test here still passed. The `pub fn` set comparison below exists so the
//!   *next* omission is caught structurally rather than by remembering.

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

/// `ViewportHandle` methods that legitimately exist on **one** platform only.
///
/// **Empty, and that is the point.** `embed_foreign` / `release_foreign` are the
/// commands' one-sided pair, and their *methods* still exist on both sides
/// (macOS's are documented no-ops) precisely because Ring 2 calls them
/// unconditionally. A non-empty entry here would mean a Ring-2 call site that has
/// to be `#[cfg]`-guarded, which is the design this whole file exists to prevent —
/// so adding one needs the reason written down beside it, like [`PLATFORM_ONLY`].
const PLATFORM_ONLY_FNS: [&str; 0] = [];

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

/// **The handle-surface gate** (P21.2 audit). The two platform handles must
/// expose the **same `pub fn` set**, modulo [`PLATFORM_ONLY_FNS`].
///
/// The `Cmd` membership gate above cannot see this, and that is exactly how
/// `voxel_volumes` slipped through: it is a handle method with **no command
/// behind it** (the save path needs the store *now*, and a channel has no reply
/// path), so it is invisible to every variant-based check. Deleting it from
/// macOS broke the macOS build and passed all five tests in this file.
///
/// The Linux-stub gate below reads win32 only, so a method added to macOS alone
/// would have escaped that too. Comparing the two real handles closes the last
/// direction.
#[test]
fn both_platform_handles_expose_the_same_methods() {
    let win = pub_fn_names(&read(WIN32));
    let mac = pub_fn_names(&read(MACOS));
    let keep = |f: &&String| !PLATFORM_ONLY_FNS.contains(&f.as_str());
    let only_win: Vec<&String> = win
        .iter()
        .filter(|f| !mac.contains(f))
        .filter(keep)
        .collect();
    let only_mac: Vec<&String> = mac
        .iter()
        .filter(|f| !win.contains(f))
        .filter(keep)
        .collect();
    assert!(
        only_win.is_empty() && only_mac.is_empty(),
        "the two `ViewportHandle` surfaces have drifted.\n  win32 only: {only_win:?}\n  macOS \
         only: {only_mac:?}\nRing 2 calls these unconditionally, so a method on one side only \
         is a build failure on the other OS. Note that a handle method needs NO command behind \
         it (`voxel_volumes` hands the shared store out directly), which is why the `Cmd` gate \
         above cannot catch this."
    );
}

/// A guard on that guard: the comparison must be over a real, populated set, not
/// two empty vectors agreeing perfectly (a renamed `ViewportHandle`, or a parser
/// that stopped matching, would otherwise make it vacuously green).
#[test]
fn the_handle_surfaces_are_actually_parsed() {
    for (label, path) in [("win32", WIN32), ("macOS", MACOS), ("the Linux stub", LIB)] {
        let fns = pub_fn_names(&read(path));
        assert!(
            fns.len() > 20,
            "only {} `pub fn` parsed out of {label} — the parser or the file moved, and every \
             comparison built on it is now vacuous",
            fns.len()
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

/// A guard on the guards: the P21.2 voxel surface really is what this batch
/// added, on every one of the three platform files.
///
/// Without this, all the tests above would still pass if the voxel tool had never
/// been wired at all — they compare the files to each other, and two files that
/// both lack a command agree perfectly.
///
/// `pub fn voxel_volumes(` is on the list because it is the one P21.2 surface with
/// **no `Cmd` behind it**: the save path needs the shared store synchronously and
/// a command channel has no reply path, so every variant-based check is blind to
/// it. It was missing from this list when the batch landed, and the audit's
/// mutation — delete it from `macos.rs` — broke the macOS build while all five
/// tests here stayed green.
#[test]
fn the_voxel_commands_reach_every_surface() {
    for (label, path) in [("win32", WIN32), ("macOS", MACOS)] {
        let src = read(path);
        for fragment in [
            "SetVoxel(VoxelSettings)",
            "ReloadVoxelStores",
            "pub fn set_voxel(",
            "pub fn reload_voxel_stores(",
            "pub fn voxel_volumes(",
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
    for fragment in [
        "pub fn set_voxel(",
        "pub fn reload_voxel_stores(",
        "pub fn voxel_volumes(",
    ] {
        assert!(
            stub.contains(fragment),
            "the Linux stub is missing `{fragment}`"
        );
    }
}

/// **The carve interaction reaches the host — win32 only, and only its
/// presence.**
///
/// Deliberately one-sided: macOS has no input, so there is nothing to mirror. It
/// is here because the audit's second mutation — deleting the *entire* win32
/// voxel interaction branch — passed every other test in this file, which meant
/// the pump gate could not see the tool it was written alongside. What it proves
/// is only that each entry point is still called from the pump; whether the calls
/// run in the right order, under the right capture state, or ever fire at runtime
/// is the hardware pass's question and not this file's (see the module docs).
#[test]
fn the_win32_pump_still_drives_the_carve_tools() {
    let src = read(WIN32);
    for fragment in [
        "ToolMode::Voxel",
        "host.begin_voxel(",
        "host.update_voxel(",
        "host.is_carving()",
        "host.finish_voxel(",
        "host.update_voxel_hover(",
        // The audit's orphaned-stroke fix: a stroke still down when the tool
        // changed has to be closed OUTSIDE the tool-gated branch, or its cuts are
        // saved with no undo entry.
        "host.settle_orphaned_carve(",
    ] {
        assert!(
            src.contains(fragment),
            "the win32 pump no longer calls `{fragment}` — the Voxel tool is wired into the \
             host but nothing drives it, which is a tool that silently does nothing (P21.2)"
        );
    }
}
