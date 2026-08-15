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
/// `pub fn voxel_volumes(` **used to be** on this list, as the one P21.2 surface
/// with no `Cmd` behind it. P23.2a hoisted the store out of the handle
/// entirely, so the method is gone from all three files and the claim it stood
/// for moved to [`the_shared_stores_are_hoisted_out_of_both_pumps`] — which
/// checks the stronger thing: that neither pump can *create* a store at all.
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

// ── the store hoist (P23.2a) ────────────────────────────────────────────────

/// Ring 2's viewport module, where the process's shared stores are created.
const RING2_VIEWPORT: &str = "editor/studio/src-tauri/src/commands/viewport.rs";
/// Ring 2's app bootstrap, where they are `manage`d.
const RING2_LIB: &str = "editor/studio/src-tauri/src/lib.rs";

/// **THE HOIST GATE** (P23.2a). Neither platform pump may CREATE a shared store.
///
/// `spawn` runs once per viewport. While it also *constructed* the carve working
/// set and the Simulate fracture map, those were per-viewport by construction —
/// invisible with one viewport and wrong with two: a carve made in the second
/// window would land in a store the save path (which has no viewport in hand)
/// never looks at, and the level would be written without it. Nothing about that
/// failure is loud; the level simply saves, missing a cave.
///
/// So the constructors belong to Ring 2 and the pumps must only ever RECEIVE
/// clones. Source text sees the regression that matters — a `shared_volumes()`
/// creeping back into a platform module — which is exactly how this would be
/// re-introduced by someone adding a third store later.
#[test]
fn the_shared_stores_are_hoisted_out_of_both_pumps() {
    for (label, path) in [("win32", WIN32), ("macOS", MACOS), ("the Linux stub", LIB)] {
        let src = read(path);
        for ctor in ["shared_volumes()", "shared_fractures()"] {
            assert!(
                !src.contains(ctor),
                "{label} calls `{ctor}` — it is MINTING a shared store instead of \
                 receiving Ring 2's. `spawn` runs once per viewport, so a store created \
                 there is per-viewport: a carve made through a second viewport would land \
                 somewhere `scene_save` never reads and the level would save without it, \
                 silently (P23.2a)."
            );
        }
        // …and the handle must not HOLD one either, which is the same claim one
        // level out: a handle field is a store whose identity depends on which
        // window is open. Scoped to the struct body — `spawn`'s parameters are
        // spelled the same way and are exactly what this batch wants to see.
        let Some(at) = src.find("pub struct ViewportHandle {") else {
            // A UNIT struct (`pub struct ViewportHandle;` — the Linux stub) has
            // no fields at all, which is the strongest possible pass. Assert it
            // really is that rather than a rename the parser missed.
            assert!(
                src.contains("pub struct ViewportHandle;"),
                "{label} has no `ViewportHandle` struct at all — was it renamed?"
            );
            continue;
        };
        let body_end = src[at..]
            .find("\n}\n")
            .map(|i| at + i)
            .unwrap_or_else(|| panic!("{label}'s `ViewportHandle` does not terminate"));
        let body = &src[at..body_end];
        for field in ["volumes:", "fractures:"] {
            assert!(
                !body.contains(field),
                "{label}'s `ViewportHandle` still carries a `{field}` — the stores are \
                 Ring 2's (`commands::SharedStores`), and a handle that holds one is a \
                 store whose identity depends on which window is open (P23.2a). Struct \
                 body was: {body}"
            );
        }
        assert!(
            body.contains("tx:"),
            "{label}'s `ViewportHandle` body parsed as {body:?} — it is not reading the \
             struct, so the field check above is vacuous"
        );
    }
}

/// The other half: both `spawn`s must TAKE the hoisted stores.
///
/// Without this, "neither pump creates one" is satisfied by a pump that has no
/// store at all — which compiles right up until the host tries to carve.
#[test]
fn both_spawns_take_the_hoisted_stores() {
    for (label, path) in [("win32", WIN32), ("macOS", MACOS), ("the Linux stub", LIB)] {
        let src = read(path);
        let at = src
            .find("pub fn spawn(")
            .unwrap_or_else(|| panic!("{label} has no `pub fn spawn`"));
        // The signature ends at the return arrow.
        let sig_end = src[at..]
            .find("-> ViewportHandle")
            .map(|i| at + i)
            .unwrap_or_else(|| panic!("{label}'s `spawn` does not return a `ViewportHandle`"));
        let sig = &src[at..sig_end];
        for ty in ["SharedVoxelVolumes", "SharedFractures"] {
            assert!(
                sig.contains(ty),
                "{label}'s `spawn` does not take a `{ty}`. The stores are created once by \
                 Ring 2 and handed in; a `spawn` that does not accept one either mints its \
                 own (see `the_shared_stores_are_hoisted_out_of_both_pumps`) or leaves the \
                 host with nothing to carve into. Signature was: {sig}"
            );
        }
    }
}

/// **Exactly one construction site, in Ring 2, managed exactly once.**
///
/// This is the "one store per process" invariant stated where it can actually be
/// broken: a second `SharedStores::default()` anywhere in the app, or a second
/// `.manage`, would give two halves of the editor two different carve stores
/// while every type still checked.
#[test]
fn ring_two_creates_the_shared_stores_exactly_once() {
    let vp = read(RING2_VIEWPORT);
    for ctor in ["shared_volumes()", "shared_fractures()"] {
        assert_eq!(
            vp.matches(ctor).count(),
            1,
            "`{ctor}` appears {} times in Ring 2's viewport module — the process gets \
             exactly ONE carve store and ONE fracture map, created inside \
             `SharedStores::default` and nowhere else (P23.2a)",
            vp.matches(ctor).count()
        );
    }
    let boot = read(RING2_LIB);
    assert_eq!(
        boot.matches(".manage(commands::SharedStores::default())")
            .count(),
        1,
        "the app bootstrap manages `SharedStores` {} times — two of them would be two \
         stores, and which one a given call site reached would depend on Tauri's state \
         map rather than on anything anyone decided",
        boot.matches(".manage(commands::SharedStores::default())")
            .count()
    );
}

/// **Every Ring-2 store reader resolves WITHOUT a viewport** (P23.2a).
///
/// The point of the hoist, asserted where it regresses: a future edit that
/// reaches the carve store or the fracture map through `ViewportState` again
/// would re-couple "what the author carved" to "which window is open". The old
/// accessors are gone, so this is a check that they do not come back.
#[test]
fn no_ring_two_reader_asks_a_viewport_for_a_store() {
    for path in [
        "editor/studio/src-tauri/src/commands/scene.rs",
        "editor/studio/src-tauri/src/commands/sim.rs",
        RING2_VIEWPORT,
    ] {
        let src = read(path);
        for banned in ["v.voxel_volumes()", "v.fracture_states()"] {
            assert!(
                !src.contains(banned),
                "{path} reaches a store through a viewport handle (`{banned}`). The save \
                 path and the Simulate publishes have no window and no camera — they read \
                 `commands::SharedStores` (P23.2a)"
            );
        }
    }
}

/// **THE LOCK RULE: `overlay_sim_carves` must not hold the document and the
/// carve store at once** (P23.2a audit).
///
/// It used to hold the *store* across the *document*, both live for a whole
/// loop — the classic two-lock deadlock shape, needing only one caller anywhere
/// to take them the other way round at the same moment. It survived review
/// because the store hung off a `ViewportHandle` and was awkward to reach; the
/// P23.2a hoist makes it one `try_state` from any command already holding the
/// document, so the shape had to go rather than be commented.
///
/// Source text is the seam: a `SimSession` needs a whole simulation and a
/// `SceneState` needs a Tauri app, so no test can drive this function. What
/// source can see — and the way this regresses — is the store lock moving back
/// above the document, or the document guard losing the block that drops it.
#[test]
fn the_sim_carve_fold_never_holds_the_document_and_the_store_at_once() {
    let src = read("editor/studio/src-tauri/src/commands/sim.rs");
    let at = src
        .find("fn overlay_sim_carves(")
        .expect("`overlay_sim_carves` was renamed");
    let end = src[at..]
        .find("\n}\n")
        .map(|i| at + i)
        .unwrap_or_else(|| panic!("`overlay_sim_carves` does not terminate"));
    let body = &src[at..end];

    let doc = body
        .find("scene.doc.lock()")
        .expect("the fold no longer locks the document — it needs the entity→asset binding");
    let store = body
        .find("store.lock()")
        .expect("the fold no longer locks the carve store — it has nothing to overlay into");
    assert!(
        doc < store,
        "`overlay_sim_carves` takes the CARVE STORE before the document. That is the \
         inverse of every other site and, held together, the two-lock deadlock shape this \
         gate exists to keep out (P23.2a audit)."
    );
    // …and the document guard must DIE before the store is taken: it is bound
    // inside a `let … = { … };` whose close brace sits between the two locks.
    // Without this, "doc first" is satisfied by holding both.
    assert!(
        body[doc..store].contains("\n    };"),
        "the document guard in `overlay_sim_carves` is not scoped to a block that ends \
         before `store.lock()`, so both locks are held at once. The rule is NO OVERLAP, \
         not an acquisition order — snapshot the bindings under the document and release \
         it (P23.2a audit)."
    );
}

/// **Primary vs All, named at every cross-module call site** (P23.2a).
///
/// The behavioural half lives in `commands/viewport.rs`'s unit tests (the
/// resolution rule over a keyed map). What no unit test can see is *which*
/// target each of the fourteen cross-module pushes chose — and the choice is
/// load-bearing in both directions:
///
/// * A **broadcast** narrowed to `Primary` leaves a second viewport streaming a
///   closed project's pages, or drawing pre-save terrain, or holding the
///   previous document's streams. All silent.
/// * The **PIE embed** widened to `All` would try to reparent ONE player window
///   into several holes at once.
///
/// Source text is the strongest available claim: a `ViewportState` cannot be
/// built without a window, so no test can drive these call sites.
#[test]
fn every_cross_module_push_names_its_target() {
    let cases: [(&str, &str, &str); 8] = [
        (
            "editor/studio/src-tauri/src/commands/assets.rs",
            "refresh_asset_index(super::Target::All)",
            "a landed asset is a fact about the content root, not one window",
        ),
        (
            "editor/studio/src-tauri/src/commands/project.rs",
            "set_content_root(super::Target::All, Some(content_root))",
            "the content root belongs to the open project",
        ),
        (
            "editor/studio/src-tauri/src/commands/project.rs",
            "set_content_root(super::Target::All, None)",
            "closing a project must stop every viewport streaming it",
        ),
        (
            "editor/studio/src-tauri/src/commands/scene.rs",
            "reload_terrain_stores(crate::commands::Target::All)",
            "the .inf_terrain on disk changed for everyone",
        ),
        (
            "editor/studio/src-tauri/src/commands/scene.rs",
            "reload_voxel_stores(crate::commands::Target::All)",
            "the terrain twin",
        ),
        (
            "editor/studio/src-tauri/src/commands/scene.rs",
            "clear_streams(crate::commands::Target::All)",
            "a document swap orphans every viewport's streams",
        ),
        (
            "editor/studio/src-tauri/src/commands/terrain.rs",
            "set_biome_palette(super::Target::All",
            "a biome vocabulary is a property of the level",
        ),
        (
            "editor/studio/src-tauri/src/commands/pie.rs",
            "embed_foreign(super::Target::Primary",
            "one player window goes into the SCENE viewport's slot",
        ),
    ];
    for (path, fragment, why) in cases {
        let src = read(path);
        assert!(
            src.contains(fragment),
            "{path} no longer contains `{fragment}` — {why} (P23.2a)"
        );
    }

    // The PIE restore pair is `Primary` for the embed's reason, and there are
    // two of each (the crash path and the explicit stop).
    let pie = read("editor/studio/src-tauri/src/commands/pie.rs");
    assert_eq!(
        pie.matches("release_foreign(super::Target::Primary)")
            .count(),
        2,
        "both PIE teardown paths (crash detection and `pie_stop`) must restore the \
         SCENE viewport explicitly"
    );
    assert_eq!(
        pie.matches("set_visible(super::Target::Primary, true)")
            .count(),
        2,
        "both PIE teardown paths must un-hide the scene viewport they hid — and only it"
    );
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
        // …and its P21.3 sibling for the three terrain brushes (height, splat,
        // biome), which had the identical hole and now share the fix. Both calls
        // must sit before the `sculpting` / `voxel` branches, which is what
        // `the_orphan_settlers_run_outside_the_tool_gated_branches` checks.
        "host.settle_orphaned_sculpt(",
    ] {
        assert!(
            src.contains(fragment),
            "the win32 pump no longer calls `{fragment}` — the Voxel tool is wired into the \
             host but nothing drives it, which is a tool that silently does nothing (P21.2)"
        );
    }
}

/// **Every gesture that survives across frames has a settler, and the pump calls
/// it** (P21.3 audit).
///
/// The other half of the settlement discipline, and the half
/// `tests/orphaned_strokes.rs` structurally cannot check: an `EngineHost` needs
/// a GPU and is `#[cfg]`-gated, so no CI machine can construct one. That file
/// gates what a settler *does* (the recorders, and the transaction door);
/// **this** gates that the settler exists at all and is wired — which is what
/// deleting `settle_orphaned_sculpt` breaks, and what nothing caught before.
///
/// Four gestures hold state across frames today: a carve stroke, a terrain
/// brush stroke, a foliage stroke, and a gizmo drag (whose state is an *undo
/// transaction* rather than a stroke object, so its settler is the generic one).
#[test]
fn every_cross_frame_gesture_has_a_settler_and_the_pump_calls_it() {
    let host = read("editor/crates/inf-viewport/src/host.rs");
    let pump = read(WIN32);
    for settler in [
        "settle_orphaned_carve",
        "settle_orphaned_sculpt",
        "settle_orphaned_foliage",
        "settle_orphaned_transaction",
        // Not a settler but the same discipline's other half: a document swap
        // ABANDONS in-flight gestures rather than committing them, because the
        // level they belonged to is gone (P21.3 audit).
        "abandon_gestures_on_document_swap",
    ] {
        assert!(
            host.contains(&format!("pub fn {settler}(")),
            "`EngineHost::{settler}` is gone — the gesture it closed can strand an edit in the \
             world with no undo entry describing it (P21.3 / the N2 ledger item)"
        );
        assert!(
            pump.contains(&format!("host.{settler}(")),
            "the win32 pump never calls `{settler}` — it exists and nothing drives it"
        );
    }
}

/// **A document swap must clear the shared carve store, not just the gesture
/// handles** (P21.3 re-audit).
///
/// The store is a **host** field, so it survives `scene_open` / `scene_new`
/// entirely. Dropping the stroke handle leaves its cut chunks in the slot and
/// they are *dirty*; when the new document binds the same `(entity, asset)` —
/// File ▸ Open on the same level, which is the gesture for *discarding* changes
/// — `retain_only` keeps the slot, `ensure` short-circuits on `is_bound`, and
/// the next Ctrl+S writes the half-carve into the `.inf_voxel` with no
/// `EditCommand` anywhere.
///
/// `voxel_dig.rs`'s `clearing_the_store_discards_an_abandoned_carve_and_saves_nothing`
/// gates what the clear *does* (nothing dirty, nothing staged, re-binding serves
/// the last-saved geometry). This gates that the abandon actually calls it —
/// which needs a GPU and a `#[cfg]`-gated module, so source text is the seam,
/// the same split the settlers use.
#[test]
fn the_document_swap_clears_the_shared_carve_store() {
    let host = read("editor/crates/inf-viewport/src/host.rs");
    let at = host
        .find("pub fn abandon_gestures_on_document_swap(")
        .expect("`abandon_gestures_on_document_swap` is gone");
    let end = host[at..]
        .find("\n    /// ")
        .map(|e| at + e)
        .unwrap_or(host.len());
    let body = &host[at..end];
    assert!(
        body.contains("volumes.clear()"),
        "`abandon_gestures_on_document_swap` drops the gesture handles but never clears the \
         shared voxel store. The chunks an abandoned stroke cut are still there and still \
         dirty, so File > Open on the same level — the gesture for THROWING CHANGES AWAY — \
         reuses them and the next save writes them into the .inf_voxel with no undo entry \
         describing them (P21.3 re-audit)."
    );
}

/// **The document-swap abandon must run BEFORE every settler** (P21.3 audit).
///
/// Order is the whole content of this one: a settler that ran first would
/// faithfully commit the *old* level's stroke into the *new* document, which is
/// precisely the outcome the abandon exists to prevent. Source text can see the
/// order of the calls, which is the way this regresses.
#[test]
fn the_document_swap_abandon_precedes_every_settler() {
    let src = read(WIN32);
    let abandon = src
        .find("host.abandon_gestures_on_document_swap(")
        .expect("the win32 pump never abandons gestures on a document swap");
    for settler in [
        "host.settle_orphaned_carve(",
        "host.settle_orphaned_sculpt(",
        "host.settle_orphaned_foliage(",
        "host.settle_orphaned_transaction(",
    ] {
        let at = src
            .find(settler)
            .unwrap_or_else(|| panic!("no `{settler}`"));
        assert!(
            abandon < at,
            "`{settler}` is called BEFORE the document-swap abandon, so a File > Open during a \
             drag commits the old level's edit into the new document"
        );
    }
}

/// **The orphan settlers must run OUTSIDE the tool-gated branches** (P21.3).
///
/// The whole point of both `settle_orphaned_*` calls is that they run when the
/// branch that would otherwise finish the stroke does not. A settler that had
/// drifted inside `if sculpting { … }` or `else if voxel { … }` would compile,
/// pass [`the_win32_pump_still_drives_the_carve_tools`] (which only asks whether
/// the fragment appears anywhere), and restore exactly the bug it was written
/// for — an edit in the document with no undo entry describing it.
///
/// Checked positionally, which is the strongest claim source text supports: both
/// calls must appear **before** the first `if sculpting {`. It cannot see
/// runtime order, and it does not pretend to; what it can see is a settler that
/// moved into a branch, which is the way this regresses.
#[test]
fn the_orphan_settlers_run_outside_the_tool_gated_branches() {
    let src = read(WIN32);
    let branch = src
        .find("if sculpting {")
        .expect("the win32 pump no longer has a `sculpting` branch — was it renamed?");
    for settler in [
        "host.settle_orphaned_carve(",
        "host.settle_orphaned_sculpt(",
        "host.settle_orphaned_foliage(",
        "host.settle_orphaned_transaction(",
    ] {
        let at = src
            .find(settler)
            .unwrap_or_else(|| panic!("the win32 pump never calls `{settler}`"));
        assert!(
            at < branch,
            "`{settler}` is called AFTER the tool-gated `if sculpting` branch begins, so it \
             cannot run for the case it exists for: a stroke stranded by a tool switch. It \
             belongs above the branches, with the document in hand (P21.3 / the N2 ledger item)."
        );
    }
}

/// **The deferred dig's safety net** (P21.4).
///
/// `SceneDoc::edit_dig` commits with the re-mesh deferred — the box cut, the
/// spline trench, the tunnel and the brush's committed stroke all land in the
/// store with their chunk versions bumped and their *meshes* left for later. That
/// is only legal because the viewport re-meshes every time it projects, and it
/// projects whenever the document version moves — which a carve does.
///
/// So two calls are now load-bearing in a way they were not before: without them
/// a dug cave is not one frame late, it is **permanently invisible**. Both are
/// pinned here, positionally where that is meaningful.
///
/// Source text is the strongest claim available (a host cannot be constructed in
/// CI — there is no window). What it can see is the call disappearing, which is
/// the way this regresses.
#[test]
fn the_projection_re_meshes_every_volume_it_projects() {
    let src = read("editor/crates/inf-viewport/src/host.rs");
    let sync = src.find("self.sync_voxels(doc);").expect(
        "the viewport projection no longer calls `sync_voxels` — a deferred dig \
                 (SceneDoc::edit_dig) would never be re-meshed and the cave would be \
                 invisible, not late",
    );
    // It runs BEFORE the entity walk that consumes the meshes it produces, and the
    // walk is really downstream (`clear` then refill), so "before" is a claim about
    // order rather than a coincidence of two offsets.
    let clear = src
        .find("self.scene.instances.clear();")
        .expect("the projection no longer clears its instance list — was it renamed?");
    assert!(
        sync < clear,
        "`sync_voxels` runs AFTER the projection has begun rebuilding its lists, so \
         a frame draws the meshes of the previous one"
    );

    // …and `sync_voxels` itself reaches the Ring-0 pass that re-meshes —
    // **unconditionally**. A body that early-returns before `sync_camera` passes a
    // bare `contains` check and produces exactly the failure this gate is named
    // for: a dug cave that is permanently invisible rather than one frame late.
    // (Verified: with the `contains` check alone, inserting `return;` at the top of
    // `sync_voxels` left this test green.)
    let body_at = src
        .find("fn sync_voxels(&mut self, doc: &SceneDoc) {")
        .expect("`sync_voxels` was renamed");
    let after = &src[body_at..];
    let end = after.find("\n    fn ").unwrap_or(after.len());
    let body = &after[..end];
    let camera = body
        .find("sync_camera(")
        .expect("`sync_voxels` no longer calls `sync_camera`, the pass that re-meshes");
    // Every `return` upstream of the re-mesh must be a **guard** — a
    // `let … else { return; }` on a poisoned lock, which is the house tolerance for
    // a thread that already panicked mid-carve. An *unconditional* one is the
    // failure this gate is named for, and a bare `contains("sync_camera")` cannot
    // tell the two apart.
    let prefix = &body[..camera];
    for (i, _) in prefix.match_indices("return") {
        let lead = &prefix[i.saturating_sub(60)..i];
        assert!(
            lead.contains("else {"),
            "`sync_voxels` returns unconditionally before it reaches `sync_camera`, \
             so a deferred dig may never be re-meshed at all — a dug cave would be \
             permanently invisible rather than one frame late"
        );
    }
}

/// **ACT FOUR OF THE PROJECTION: the sim → render voxel fold** (P21.4).
///
/// A `voxel.*` node writes the *simulation's* volume map, which is deliberately
/// not the camera-paged store either host draws from. Both hosts therefore owe a
/// fold, and neither is covered by the pump-mirror family's `Cmd`-based checks:
/// the player's lives in `render::sync_voxel_store` and the **editor's lives in
/// Ring 2** (`commands/sim.rs`), because the viewport pump has no reference to a
/// `SimSession` to fold from. Two different call sites, one rule, and nothing
/// structural connecting them — which is how one of them gets deleted.
///
/// The failure is silent in the way this whole batch has been: a host without the
/// fold gives the dug floor a collider, lets gameplay stand on it, and keeps
/// drawing the rock. Source text is the strongest claim available (neither host
/// can be constructed in CI); what it can see is a call site disappearing.
#[test]
fn both_hosts_fold_the_sims_carves_into_their_render_store() {
    // The player: inside the CPU half of its voxel sync, and **after** the camera
    // pass, which is what keeps the pin-free re-copy correct.
    let player = read("runtime/inf-player/src/render.rs");
    let sync = player
        .find("pub fn sync_voxel_store(")
        .expect("the player's voxel sync was renamed");
    let body = &player[sync..];
    let camera = body
        .find("sync_camera(")
        .expect("the player's voxel sync no longer pages against the camera");
    let overlay = body.find("overlay_sim(").expect(
        "the SHIPPED PLAYER no longer folds the sim's carves into its render store — \
         a game that digs would draw the rock it removed",
    );
    assert!(
        overlay > camera,
        "the player's overlay runs BEFORE the camera pass, so residency can page \
         the asset's pre-carve bytes back over a carve with nothing to undo it"
    );
    assert!(
        body[overlay..].contains("resync("),
        "the player's overlay never re-meshes what it copied, so the carve is in \
         the field and not on the screen"
    );

    // The editor: from Ring 2, after the session ticks. Both tick paths, because
    // `sim_tick` (free-running) and `sim_step_fixed` (Step) are separate commands
    // and a fold on only one of them is a debugger that shows a different world.
    let ring2 = read("editor/studio/src-tauri/src/commands/sim.rs");
    assert!(
        ring2.contains("fn overlay_sim_carves("),
        "the EDITOR no longer folds the sim's carves into the viewport store"
    );
    assert_eq!(
        ring2.matches("overlay_sim_carves(&app").count(),
        2,
        "the editor's fold is called from {} of its two tick commands (`sim_tick` \
         and `sim_step_fixed`) — Step and Play would show different worlds",
        ring2.matches("overlay_sim_carves(&app").count()
    );
    assert!(
        ring2.contains("store.overlay_sim("),
        "the editor's fold no longer reaches the Ring-0 rule"
    );
}

// ── the spawn seam (P22.3) ──────────────────────────────────────────────────

/// The `let` bindings inside `pub fn spawn(…)`, in declaration order.
///
/// Tuple destructuring (`let (tx, rx) = …`) contributes both names; everything
/// else contributes one. Types, `mut` and the initialiser are dropped — the set
/// is what matters, not how each was produced.
fn spawn_bindings(source: &str) -> Vec<String> {
    let start = source
        .find("pub fn spawn(")
        .expect("both platform modules have a `pub fn spawn`");
    // The function ends at the next item at column 0 after it.
    let end = source[start..]
        .find("\n}\n")
        .map(|i| start + i)
        .unwrap_or(source.len());
    let mut out = Vec::new();
    for line in source[start..end].lines() {
        // FUNCTION SCOPE ONLY (exactly four spaces of indent). The shared stores
        // the handle hands to Ring 2 are declared there; a `let` nested inside
        // macOS's `unsafe` layer-construction block is a local with no twin to
        // have, and including it would make the allow-list a list of macOS's
        // implementation details.
        if !line.starts_with("    ") || line.starts_with("     ") {
            continue;
        }
        let t = line.trim();
        let Some(rest) = t.strip_prefix("let ") else {
            continue;
        };
        let rest = rest.strip_prefix("mut ").unwrap_or(rest);
        let Some(lhs) = rest.split('=').next() else {
            continue;
        };
        let lhs = lhs.split(':').next().unwrap_or(lhs).trim();
        if let Some(tuple) = lhs.strip_prefix('(').and_then(|s| s.strip_suffix(')')) {
            out.extend(
                tuple
                    .split(',')
                    .map(|n| n.trim().to_string())
                    .filter(|n| !n.is_empty()),
            );
        } else if lhs != "_"
            && !lhs.is_empty()
            && lhs.chars().all(|c| c.is_alphanumeric() || c == '_')
        {
            // `let _ = sink;` is a discard, not a binding — macOS has one because
            // its input is unwired, and it names nothing that could have a twin.
            out.push(lhs.to_string());
        }
    }
    out
}

/// Field names of every `ViewportHandle { … }` struct literal in `source`.
fn handle_literal_fields(source: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut from = 0;
    while let Some(i) = source[from..].find("ViewportHandle {") {
        let at = from + i;
        // `-> ViewportHandle {` is a return type followed by the function's OWN
        // brace, and `pub struct ViewportHandle {` is the definition. Both contain
        // this text and neither is a construction; matching them slurps an
        // unrelated brace span and the comparison becomes noise.
        let before = source[..at].trim_end();
        if before.ends_with("->") || before.ends_with("struct") || before.ends_with("pub struct") {
            from = at + "ViewportHandle {".len();
            continue;
        }
        let open = at + "ViewportHandle {".len();
        let Some(close) = source[open..].find('}') else {
            break;
        };
        for f in source[open..open + close].split(',') {
            let name = f.split(':').next().unwrap_or(f).trim();
            if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                out.push(name.to_string());
            }
        }
        from = open + close;
    }
    out.sort();
    out.dedup();
    out
}

/// **THE SPAWN-BINDING MIRROR — the third instance of this class gets a gate.**
///
/// The two platform `spawn` functions each create the shared stores the handle
/// hands to Ring 2 (`volumes`, and since P22.3 `fractures`), clone them for the
/// render thread, and construct a `ViewportHandle` from them. A store added to
/// win32 alone is not a missing feature: it is a macOS build that does not
/// compile — and it has now happened three times.
///
/// * P20.4: `set_water`/`set_water_hints` on win32 only → macOS CI red, fixed in
///   a follow-up commit. That is what this file was written for.
/// * P21.2: `pub fn voxel_volumes` missing from the fragment list → deleting it
///   broke the build while every test here passed. That is what the `pub fn` set
///   comparison was added for.
/// * P22.3: `let fractures` / `let host_fractures` added to win32's `spawn` and
///   not to macOS's → **E0425 × 4, macOS clippy leg red**. Neither existing check
///   could see it: a local binding is not a `pub fn` and not a `Cmd` variant, and
///   the fragment lists compare text that was identical on both sides (the struct
///   FIELD was mirrored; the binding that fills it was not).
///
/// So this compares the two sets directly. win32's bindings must all exist on
/// macOS; macOS may have more (it builds a `CAMetalLayer` and win32 does not), and
/// those extras are listed rather than waved through.
#[test]
fn every_win32_spawn_binding_has_a_macos_twin() {
    let win = spawn_bindings(&read(WIN32));
    let mac = spawn_bindings(&read(MACOS));

    // Sanity: the parser found the real thing on both sides, or the comparison
    // below is between two empty lists (which agree).
    for (label, set) in [("win32", &win), ("macOS", &mac)] {
        assert!(
            set.contains(&"tx".to_string()) && set.contains(&"rx".to_string()),
            "the {label} spawn-binding parser found {set:?} — it is not reading \
             `pub fn spawn`"
        );
    }

    let missing: Vec<&String> = win.iter().filter(|b| !mac.contains(b)).collect();
    assert!(
        missing.is_empty(),
        "win32's `spawn` binds {missing:?} and macOS's does not — every shared \
         store the handle hands to Ring 2 is created here, so a binding on one \
         side only is a build that does not compile on the other. (win32: {win:?}; \
         macOS: {mac:?})"
    );

    // macOS's extras, named. A new one appearing here is a prompt to decide
    // whether it is genuinely platform-specific rather than a win32 omission.
    let extra: Vec<&String> = mac.iter().filter(|b| !win.contains(b)).collect();
    let allowed = ["layer_ptr", "scale"];
    let unexpected: Vec<&&String> = extra
        .iter()
        .filter(|b| !allowed.contains(&b.as_str()))
        .collect();
    assert!(
        unexpected.is_empty(),
        "macOS's `spawn` binds {unexpected:?}, which win32 does not and which is \
         not in the platform-specific allow-list — either win32 is missing it, or \
         add it to `allowed` deliberately"
    );
}

/// The same claim one level out: both platforms must construct the handle from
/// the same FIELDS. (This one would already have passed the P22.3 miss — the
/// fields were mirrored and the bindings were not — which is exactly why the test
/// above exists beside it rather than instead of it.)
#[test]
fn both_platforms_construct_the_handle_from_the_same_fields() {
    let win = handle_literal_fields(&read(WIN32));
    let mac = handle_literal_fields(&read(MACOS));
    assert!(
        win.contains(&"tx".to_string()),
        "the handle-literal parser found {win:?} — it is not reading the struct \
         literals"
    );
    assert_eq!(
        win, mac,
        "the two platforms construct `ViewportHandle` from different fields — one \
         of them will not compile"
    );
}
