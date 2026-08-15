//! **A device-loss rebuild must forget every answer that was about the dead
//! device** (Hardening D).
//!
//! # Why this is a source gate and not a behavioural one
//!
//! There is no way to make a device be lost from a test. `GpuContext::is_lost` is
//! a wgpu callback the driver fires on a TDR or a removed adapter; the branch
//! under gate here runs on that signal alone, needs a real window to rebuild
//! against (`EngineHost` takes a `SurfaceTarget`), and its whole failure mode is
//! *silence* — the viewport keeps drawing, untextured and at crate-default
//! settings, for the rest of the session. So the enforcement that exists is the
//! one Wave C stated: **where a fix's correctness is "the code still does this",
//! only a source pin is enforcement.**
//!
//! # What is pinned, and why exactly these fields
//!
//! Each one is a *memo of something already pushed into the old renderer* that
//! gates an early return, so a fresh renderer that never receives the push is
//! indistinguishable from one that already had it:
//!
//! | field | the early return it gates | what the author sees |
//! |---|---|---|
//! | `vt_level_key` | `sync_vt_bindings` | every virtual-textured surface untextured |
//! | `applied_render` | `apply_render_settings` | the level's post/exposure block silently defaulted |
//! | `synced_version` | `sync_from_doc` | the new renderer holds no scene |
//! | `render_tier` | the cached tier probe | a clamp computed for the OLD adapter |
//! | `render_caps` | the cached caps probe | the same, for the occlusion floor |
//!
//! The list is an **allowlist of what must be reset**, not a ban list of what
//! must not be kept — the P22 rule, because a ban list only ever enumerates what
//! somebody thought of. When a new memo field joins the host, this gate does not
//! notice it; the doc comment on `reset_device_scoped_state` is where the rule
//! lives, and this file is what stops the *existing* reset from quietly rotting.

/// The host's source text. Line endings are normalized before any search — the
/// P22 CRLF law, met by every gate in this repo that reads a `.rs`.
fn host_src() -> String {
    include_str!("../src/host.rs").replace("\r\n", "\n")
}

/// The body of `fn <name>` up to the closing brace at its own (4-space)
/// indentation.
fn fn_body(src: &str, name: &str) -> String {
    let decl = format!("fn {name}(");
    let start = src
        .find(&decl)
        .unwrap_or_else(|| panic!("`{decl}` occurs nowhere — was it renamed?"));
    let body = &src[start..];
    let end = body
        .find("\n    }\n")
        .unwrap_or_else(|| panic!("`{decl}` does not terminate at method indentation"));
    body[..end].to_string()
}

#[test]
fn the_device_loss_branch_resets_the_dead_device_s_memos() {
    let src = host_src();
    let render_frame = fn_body(&src, "render_frame");
    assert!(
        render_frame.contains("if self.gpu.is_lost() {"),
        "the device-loss branch is not where this gate expects it — `render_frame` \
         no longer tests `self.gpu.is_lost()`"
    );
    assert!(
        render_frame.contains("self.reset_device_scoped_state("),
        "the device-loss branch rebuilds the GPU stack without resetting the memos \
         of what was pushed into the DEAD renderer. A fresh `EngineRenderer` starts \
         with no virtual texture, default settings and `ViewMode::Lit`; every memo \
         left standing gates an early return that stops the re-push."
    );
    assert!(
        render_frame.contains("self.picker = Picker::new(&self.gpu);"),
        "the picker's own device-scoped resources must still be rebuilt (H1)"
    );
    assert!(
        render_frame.contains("self.chain.release();"),
        "the dead swapchain must be dropped BEFORE `build_gpu_stack` creates a \
         second Instance + Surface for the same window"
    );
}

#[test]
fn the_reset_names_every_memo_that_gates_a_re_push() {
    let body = fn_body(&host_src(), "reset_device_scoped_state");
    for (field, consequence) in [
        (
            "self.vt_level_key = Default::default();",
            "`sync_vt_bindings` returns early on an unchanged key, so every \
             virtual-textured surface renders untextured for the rest of the session",
        ),
        (
            "self.applied_render = None;",
            "`apply_render_settings` returns early on an unchanged block, so the \
             level's authored post/exposure/vgeom/VSM block is never pushed",
        ),
        (
            "self.synced_version = None;",
            "the projection is version-gated and the fresh renderer holds no scene",
        ),
        (
            "self.render_tier = None;",
            "the cached tier is a probe of the OLD adapter",
        ),
        (
            "self.render_caps = None;",
            "the cached caps are a probe of the OLD adapter",
        ),
    ] {
        assert!(
            body.contains(field),
            "`reset_device_scoped_state` no longer clears `{field}` — {consequence}"
        );
    }
    assert!(
        body.contains("set_view_mode(view_mode)"),
        "the shading mode lives only in the renderer, so it must be re-pushed \
         (a fresh renderer starts at `Lit`) rather than cleared"
    );
}
