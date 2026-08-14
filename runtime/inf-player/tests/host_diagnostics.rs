//! **The lines nothing logged** (P28.5) — the last of the "a line nothing logs"
//! class, closed and armed.
//!
//! Three public functions in `inf-render` each carry the doc comment *"the one
//! line a host logs about …"*:
//!
//! | function | landed | logged by a host |
//! |---|---|---|
//! | `EngineRenderer::vsm_summary` | P27.1 | P27.5 — the remainder was named and closed |
//! | `EngineRenderer::vt_summary` | P26.5 | **never**, until this batch |
//! | `EngineRenderer::stream_summary` | P28.3 | **never** — the P28.3 audit gave it a *gate* reader and said so |
//! | `EngineRenderer::predict_summary` | P28.5 | this batch, and it is the first reader `Prediction::turn`, `Prediction::clamped` and `CameraHistory::refused` have ever had |
//!
//! A counter with no reader is not a defect and this file does not call it one.
//! What it is, is a claim nothing can check: *"a host that wants to see the
//! clamp bind can"* is true of any `pub` field ever written. So the plan does
//! not end with the class open.
//!
//! # What each arm is built to falsify
//!
//! * **The host really calls them.** A source scan over `window.rs`'s
//!   `log_stream_stats` — read as a **scope**, not as a ban list (the P23 law:
//!   a ban enumerates what you thought of). Deleting any one of the four calls
//!   fails it; moving one outside the function fails it.
//! * **The wrappers really delegate.** `PlayerRenderHost`'s three new methods
//!   are named as function items here, so a rename or a removal is a compile
//!   error in this file rather than a silent loss of the line.
//! * **The line says what it counts.** On a real renderer, with a committed
//!   camera history, the predictor's line names its four numbers — and is
//!   `None` before any host commits, which is the editor viewport's permanent
//!   state and must not read as "the predictor is broken".

use inf_render::{EngineRenderer, GpuContext, RenderView};

/// `window.rs`, as source. The P22 law is why this is safe to search: `.rs` is
/// `text eol=lf` in `.gitattributes`, so a Windows checkout does not change the
/// bytes a `contains` looks for.
const WINDOW_RS: &str = include_str!("../src/window.rs");

/// The body of one `fn` in [`WINDOW_RS`], by brace matching from its signature.
///
/// A scope and not a line range: an arm that searched the whole file would be
/// satisfied by the four calls sitting anywhere at all, including in a comment
/// or in a function nothing calls.
fn fn_body(src: &str, signature: &str) -> String {
    let start = src
        .find(signature)
        .unwrap_or_else(|| panic!("`{signature}` is not in window.rs"));
    let open = src[start..]
        .find('{')
        .expect("the function has a body")
        + start;
    let mut depth = 0usize;
    for (i, c) in src[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return src[open..=open + i].to_string();
                }
            }
            _ => {}
        }
    }
    panic!("`{signature}` never closes");
}

/// **The shipped host logs all four streaming lines**, and it does it inside
/// the one function that owns the once-a-second cadence.
#[test]
fn the_shipped_host_logs_every_streaming_line_it_has() {
    let body = fn_body(WINDOW_RS, "fn log_stream_stats(");
    for line in [
        "vsm_summary",
        "vt_summary",
        "stream_summary",
        "predict_summary",
    ] {
        // **The EMITTING spelling, not the mention.** The first draft of this
        // arm asked for `host.{line}()` anywhere in the scope and was killed by
        // its own mutation: three of the four are named twice — once in the
        // early-return guard as `.is_some()` and once where the line is
        // actually logged — so deleting the *log* left the guard's mention
        // behind and the arm stayed green. A gate must aim at the thing it
        // names (the P23 law), and the thing this one names is the emit.
        let emit = format!("and_then(|l| l.host.{line}())");
        assert!(
            body.contains(&emit),
            "`log_stream_stats` does not LOG `{line}` — it may still mention it \
             in the guard, which is exactly the shape this arm was rewritten to \
             see. The line it produces would be a line nothing logs, which is \
             the class P28.5 closed"
        );
        assert!(
            body.matches(&emit).count() == 1,
            "`{line}` is logged {} times — a duplicated emit means the once-a-\
             second cadence prints it twice",
            body.matches(&emit).count()
        );
    }
    // The gate itself must be falsifiable: the scope really is a scope, and a
    // spelling that exists elsewhere in the file is not enough.
    assert!(
        body.len() < WINDOW_RS.len() / 4,
        "`fn_body` returned most of the file, so this arm is a whole-file scan \
         wearing a scope's clothes"
    );
    assert!(
        WINDOW_RS.contains("fn frame(") && !body.contains("fn frame("),
        "the extracted scope leaked into the next function"
    );
    // …and each line is emitted rather than merely mentioned: the guard that
    // decides whether the whole block runs at all also has to name them, or a
    // level with only virtual textures would return before reaching them.
    assert!(
        body.contains("let paging =") && body.contains("if !terrain && !cells && !shadows"),
        "the early-return guard does not know about the three new lines"
    );
}

/// **The host's wrappers exist and are the renderer's**, named as items so a
/// rename is a compile error here.
#[test]
fn the_player_host_exposes_the_three_lines() {
    type Line = fn(&inf_player::render::PlayerRenderHost) -> Option<String>;
    let wrappers: [(&str, Line); 4] = [
        ("vsm", inf_player::render::PlayerRenderHost::vsm_summary),
        ("vt", inf_player::render::PlayerRenderHost::vt_summary),
        (
            "stream",
            inf_player::render::PlayerRenderHost::stream_summary,
        ),
        (
            "predict",
            inf_player::render::PlayerRenderHost::predict_summary,
        ),
    ];
    assert_eq!(wrappers.len(), 4);
    // The names are distinct function items — four pointers, four functions.
    for (i, (a, fa)) in wrappers.iter().enumerate() {
        for (b, fb) in wrappers.iter().skip(i + 1) {
            assert!(
                !std::ptr::fn_addr_eq(*fa, *fb),
                "`{a}` and `{b}` are the same function"
            );
        }
    }
}

/// **The predictor's line is `None` before a host commits, and names its four
/// numbers after.**
///
/// The `None` half is the one that matters and it is not a courtesy: an empty
/// history is the predictor's real enable flag, and the editor viewport's
/// permanent state. A host that printed a line of zeros there would be saying
/// "the predictor is running and finding nothing", which is false.
#[test]
fn the_predictors_line_appears_only_once_a_host_commits_a_pose() {
    let Ok(gpu) = GpuContext::headless() else {
        eprintln!("SKIP host_diagnostics: no GPU adapter");
        return;
    };
    let mut r = EngineRenderer::new(&gpu, inf_render::HEADLESS_FORMAT);
    assert!(
        r.predict_summary().is_none(),
        "a renderer nobody committed a camera to produced a predictor line"
    );

    let view = |tick: u64| RenderView {
        origin: inf_math::FloatingOrigin::new(glam::DVec3::ZERO),
        eye_world: glam::DVec3::new(0.1 * tick as f64, 1.6, 0.0),
        forward: glam::Vec3::new(
            inf_math::psin64(0.05 * tick as f64) as f32,
            0.0,
            -inf_math::pcos64(0.05 * tick as f64) as f32,
        )
        .normalize(),
        up: glam::Vec3::Y,
        fov_y: 60_f32.to_radians(),
        near: 0.1,
        width: 64,
        height: 64,
        ortho: None,
    };
    for tick in 0..6u64 {
        assert!(r.commit_camera(tick, &view(tick)));
    }
    // A repeat of the newest tick — the refusal the counter exists for, and the
    // one that says "this host is wired to the render loop".
    assert!(!r.commit_camera(5, &view(5)));

    let line = r.predict_summary().expect("a committed history has a line");
    println!("{line}");
    for token in ["predict:", "6 committed poses", "(1 refused)", "horizon", "lane"] {
        assert!(line.contains(token), "{token:?} missing from {line}");
    }
    // The shipped horizon is zero, so the reckoner turns by nothing — and the
    // line says so rather than omitting the field. That is the lead-time
    // ruling, visible in a log.
    assert_eq!(inf_render::DEFAULT_PREDICT_HORIZON_TICKS, 0);
    assert!(line.contains("turn 0.00 deg"), "{line}");
    assert!(!line.contains("CLAMPED"), "{line}");

    // …and at the ROADMAP's refuted lead the same line reports a real turn,
    // which is what says the field is read and not printed as a constant.
    let mut s = *r.settings();
    s.stream.predict.horizon_ticks = inf_render::ROADMAP_PREDICT_HORIZON_TICKS;
    r.try_set_settings(s).expect("a legal horizon");
    let led = r.predict_summary().expect("still a history");
    println!("{led}");
    assert!(!led.contains("turn 0.00 deg"), "{led}");

    // **A cleared history keeps its line, and that is deliberate.** A camera cut
    // empties the window, but `refused` is a *session* fact — it is the number
    // that says "this host is wired to the render loop rather than to the fixed
    // step", and a host whose every push is refused would otherwise produce
    // exactly the silence that means "nothing here commits a pose". The one
    // number that explains the silence may not be silent itself.
    r.reset_camera_history();
    let after = r
        .predict_summary()
        .expect("a host that has refused a push keeps its line");
    println!("{after}");
    assert!(after.contains("0 committed poses (1 refused)"), "{after}");
    assert!(after.contains("no prediction"), "{after}");
}

/// …and the other half of that ruling: a renderer that has **never been offered
/// a pose at all** produces no line, which is the editor viewport's permanent
/// and correct state.
///
/// The pair is the arm. Without this one, "the line appears when a host commits"
/// is satisfied by a line that always appears.
#[test]
fn a_renderer_no_host_ever_offered_a_pose_produces_no_predictor_line() {
    let Ok(gpu) = GpuContext::headless() else {
        eprintln!("SKIP host_diagnostics: no GPU adapter");
        return;
    };
    let r = EngineRenderer::new(&gpu, inf_render::HEADLESS_FORMAT);
    assert!(r.camera_history().is_empty());
    assert_eq!(r.camera_history().refused(), 0);
    assert!(r.predict_summary().is_none());
}
