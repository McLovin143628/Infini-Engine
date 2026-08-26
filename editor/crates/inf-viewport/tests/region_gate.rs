//! **The cutout survives the three things that take it away** (UX2).
//!
//! A window region is not a property the child keeps for you. It is discarded
//! by a resize, it is expressed in coordinates a resize moves, and it is
//! meaningless while an embedded PIE window occupies the slot. Each of those is
//! handled at the point it happens in the viewport loop, and each failure looks
//! the same from outside: the 3D view goes black behind a menu again, or a
//! chunk of it disappears and stays gone. Neither throws, neither logs.
//!
//! # Why a source gate
//!
//! There is nothing to assert at runtime: CI has no window, no adapter and no
//! compositor, and `win32.rs` is `#[cfg(windows)]` besides. The mechanism's
//! *geometry* is unit-tested next to it (`child_local_cutouts`, four arms) and
//! its *cost* is measured by the `#[ignore]`d `region_present_cost`; what is
//! left is the wiring, which is a claim about where a call sits. So the pin is
//! on the source, and — the P23 law — it reads a SCOPE rather than a spelling
//! anywhere in the file: a call in the wrong block satisfies a `contains` and
//! nothing else.
//!
//! `include_str!` is safe on a Windows checkout because `.rs` carries
//! `text eol=lf` in `.gitattributes` (P22.4); the CRLF strip is kept anyway,
//! since a locally-created file has whatever the editor wrote (the fourth time
//! this has bitten a source gate).

const WIN32: &str = include_str!("../src/win32.rs");

/// The source between `start` and the next `end` after it, both required.
fn scope(start: &str, end: &str) -> String {
    let src = WIN32.replace("\r\n", "\n");
    let from = src
        .find(start)
        .unwrap_or_else(|| panic!("win32.rs must contain `{start}`"));
    let rest = &src[from..];
    let to = rest
        .find(end)
        .unwrap_or_else(|| panic!("`{start}` must be followed by `{end}`"));
    rest[..to].to_string()
}

/// **A resize rebuilds the region.** `SetWindowPos` discards it, and the region
/// is stored in child coordinates, so the origin it was built against has just
/// moved. Dragging a splitter with a menu open either restores the blackout the
/// wave removed or clips the child to a rectangle it no longer occupies.
#[test]
fn a_resize_reapplies_the_cutout() {
    let block = scope("if let Some(r) = latest_rect {", "host.resize(");
    assert!(
        block.contains("apply_window_region("),
        "the SetRect block must rebuild the window region against the new \
         rectangle. Block:\n{block}"
    );
}

/// The same block must pass the NEW rectangle, not the stale `last_rect` it
/// just overwrote — the whole reason the subtraction happens on this side is
/// that the origin used is the one the child was just positioned at.
#[test]
fn the_resize_reapplies_against_the_new_rectangle() {
    let block = scope("if let Some(r) = latest_rect {", "host.resize(");
    assert!(
        block.contains("apply_window_region(hwnd, &cutouts, r)"),
        "the re-application must use `r`, the rectangle just applied. Block:\n{block}"
    );
}

/// **The PIE skip, on both commands.** While a player window is embedded our
/// child is hidden behind it; `SetVisible` has skipped it since P9.4 and the
/// region path must skip it for the same reason — shaping a hidden window only
/// leaves a stale region for `ReleaseForeign` to restore with.
///
/// Asserted together because the failure is asymmetric by nature: the two arms
/// are separate `match` arms and only one of them had the skip before this wave.
#[test]
fn neither_visibility_nor_region_is_pushed_while_pie_is_embedded() {
    for (arm, next) in [
        ("Ok(Cmd::SetVisible(v)) => {", "Ok(Cmd::SetRegion"),
        ("Ok(Cmd::SetRegion(rects)) => {", "Ok(Cmd::Drop"),
    ] {
        let block = scope(arm, next);
        assert!(
            block.contains("embedded.is_none()"),
            "the {arm} arm must skip while a PIE window is embedded. Arm:\n{block}"
        );
    }
}

/// Releasing an embedded PIE window puts the cutout back. A menu opened while
/// the preview was up would otherwise be occluded the instant the player window
/// went away — the same bug, one frame later.
#[test]
fn releasing_an_embedded_window_restores_the_cutout() {
    let block = scope("Ok(Cmd::ReleaseForeign) => {", "Ok(Cmd::Destroy)");
    assert!(
        block.contains("apply_window_region("),
        "ReleaseForeign must restore the region the shell is holding. Arm:\n{block}"
    );
}

/// **No cutouts is the released region.** This is the state every path has to
/// be able to get back to: a menu closing, an overlay releasing, the fallback
/// full-hide. A `SetWindowRgn` that is only ever called with a region leaves a
/// permanent hole in the viewport the first time one is applied.
#[test]
fn an_empty_cutout_set_releases_the_region() {
    let block = scope("fn apply_window_region(", "\nfn region_hole(");
    assert!(
        block.contains("SetWindowRgn(hwnd, None, true)"),
        "with no cutouts the region must be RELEASED, not left. Fn:\n{block}"
    );
}

/// **The GDI ownership rule.** `SetWindowRgn` takes ownership of the region on
/// success and does not on failure, so the failure path has to delete it — a
/// leaked `HRGN` per menu open is a handle leak against a 10 000-object ceiling,
/// and it degrades the whole process, not just the viewport. The intermediate
/// rectangles combined into the region are the same story.
#[test]
fn a_region_that_was_not_handed_over_is_deleted() {
    let block = scope("fn apply_window_region(", "\nfn region_hole(");
    assert!(
        block.contains("if SetWindowRgn(hwnd, Some(full), true) == 0"),
        "the failure of SetWindowRgn must be checked — it is the case where the \
         system did NOT take the region. Fn:\n{block}"
    );
    assert_eq!(
        block.matches("DeleteObject(").count(),
        2,
        "both the per-cutout rectangle and the not-handed-over region must be \
         deleted. Fn:\n{block}"
    );
}
