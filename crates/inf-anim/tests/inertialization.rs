//! **Inertialization is cheaper than the cross-fade it replaces** — the wall
//! clock half (P29.2 clause 5).
//!
//! §13's catalogue amendment makes two claims about inertialization and this file
//! owns one of them: *"simultaneously cheaper and smoother than the cross-fade"*.
//! The **smoothness** half is measured in `inertialize.rs`'s own arms (peak
//! second difference of the rendered angle, at the tail of a transition and over
//! the whole of one). The **cost** half has two legs here:
//!
//! * a **functional** leg — the count of state evaluations, which is a property of
//!   the program and asserts on every platform;
//! * a **timing** leg — seconds, which does not.
//!
//! # The paravirtual-runner law
//!
//! macOS CI runs paravirtualised and cannot support a *ratio* between two
//! microbenchmarks: the same pair of runs has been seen inverted there on
//! unrelated work. The timing leg therefore **reports and does not assert** on
//! macOS, by name, and every timing figure is the **minimum** of several rounds —
//! a minimum is the round least disturbed by the scheduler, where a mean is the
//! round most disturbed by it.

use std::cell::Cell;
use std::time::Instant;

use glam::{Mat4, Quat, Vec3};
use inf_anim::{
    AnimClip, ClipRef, Interpolation, Joint, JointTransform, JointTrack, PoseBlender, QuatTrack,
    Skeleton, SmContext, SmRuntime, SmState, SmTransition, StateMachine,
};
use inf_anim::state_machine::CmpOp;

/// How many joints the benchmark rig has. A real humanoid is 60–90; the point of
/// the number is that clip sampling has to dominate the loop, or the measurement
/// is about the state machine's bookkeeping rather than about how many clips it
/// sampled.
const JOINTS: usize = 64;

/// Keys per rotation track — enough that `locate` does real work.
const KEYS: usize = 24;

/// A straight chain of `JOINTS` joints.
fn rig() -> Skeleton {
    let mut joints = Vec::with_capacity(JOINTS);
    let mut global = Mat4::IDENTITY;
    for i in 0..JOINTS {
        let local = JointTransform::from_trs(
            if i == 0 { Vec3::ZERO } else { Vec3::Y * 0.1 },
            Quat::IDENTITY,
            Vec3::ONE,
        );
        global *= local.to_mat4();
        joints.push(Joint {
            name: format!("j{i}"),
            parent: if i == 0 { None } else { Some(i as u16 - 1) },
            inverse_bind: global.inverse().to_cols_array(),
            local_bind: local,
        });
    }
    Skeleton::new(joints).expect("a straight chain is a valid rig")
}

/// A clip rotating every joint through `KEYS` keys, offset by `phase` so the two
/// clips are genuinely different poses.
fn dense_clip(name: &str, phase: f32) -> AnimClip {
    // `pslerp` from identity toward a half-turn: an exact angle with no `sin`, so
    // the fixture obeys the same portability rule the code under test does.
    let half = Quat::from_xyzw(0.0, 0.0, 1.0, 0.0);
    let tracks = (0..JOINTS)
        .map(|j| {
            let times: Vec<f32> = (0..KEYS).map(|k| k as f32 / (KEYS - 1) as f32).collect();
            let values: Vec<[f32; 4]> = (0..KEYS)
                .map(|k| {
                    let a = ((k as f32 * 7.0 + j as f32 * 3.0) % 60.0 + phase) / 180.0;
                    inf_math::pslerp(Quat::IDENTITY, half, a).to_array()
                })
                .collect();
            let mut t = JointTrack::new(j as u16);
            t.rotation = Some(QuatTrack::new(times, values, Interpolation::Linear));
            t
        })
        .collect();
    AnimClip::new(name, tracks)
}

fn machine() -> StateMachine {
    StateMachine {
        states: vec![SmState::clip("a", [1; 16]), SmState::clip("b", [2; 16])],
        transitions: vec![
            SmTransition::on(0, 1, 0.30, "go", CmpOp::Gt, 0.5),
            SmTransition::on(1, 0, 0.30, "go", CmpOp::Lt, 0.5),
        ],
        entry: 0,
        ..Default::default()
    }
}

/// Run `steps` fixed steps, toggling the transition every `period` steps so the
/// machine spends most of its time mid-blend — which is the only regime in which
/// the two mechanisms cost different amounts.
fn run(blender: &mut PoseBlender, steps: usize, period: usize) -> f32 {
    let sk = rig();
    let (a, b) = (dense_clip("a", 0.0), dense_clip("b", 90.0));
    let clips = |c: ClipRef| -> Option<&AnimClip> {
        match c[0] {
            1 => Some(&a),
            2 => Some(&b),
            _ => None,
        }
    };
    let sm = machine();
    let mut rt = SmRuntime::default();
    let go = Cell::new(0.0f64);
    let vars = |n: &str| (n == "go").then(|| go.get());
    let ctx = SmContext::new(&vars);
    // A value derived from every pose, so nothing can be optimised away.
    let mut checksum = 0.0f32;
    for k in 0..steps {
        if k % period == 0 {
            go.set(if go.get() > 0.5 { 0.0 } else { 1.0 });
        }
        let (pose, _) = blender.step(&sm, &mut rt, &sk, &clips, &ctx, 1.0 / 60.0);
        checksum += pose.locals[JOINTS / 2].rotation[3];
    }
    checksum
}

/// **The functional leg: inertialization evaluates strictly fewer states.**
///
/// Asserts everywhere. This is arithmetic on a counter, not a clock — the
/// paravirtual law has nothing to say about it, and it is the claim that actually
/// matters, because "cheaper" here is a statement about the algorithm.
#[test]
fn inertialization_samples_fewer_states_than_the_cross_fade() {
    const STEPS: usize = 600;
    const PERIOD: usize = 40;
    let mut inert = PoseBlender::new();
    let mut faded = PoseBlender::cross_fade();
    run(&mut inert, STEPS, PERIOD);
    run(&mut faded, STEPS, PERIOD);
    let (i, f) = (inert.evaluations(), faded.evaluations());
    assert_eq!(i, STEPS as u64, "one state per step, always");
    assert!(
        f > i,
        "the cross-fade evaluated {f} states against inertialization's {i}"
    );
    // 0.30 s at 60 Hz is 18 frames of two evaluations per 40-step cycle, so the
    // cross-fade pays about 45% more. The bound is deliberately loose — the exact
    // ratio is a function of the fixture's period, and the claim is "materially
    // more", not a number.
    assert!(
        f >= i + (STEPS as u64 * 3) / 10,
        "the cross-fade's extra evaluations ({}) were not the second state it is \
         supposed to be paying for",
        f - i
    );
}

/// **The timing leg.** Reports on every platform; asserts everywhere except
/// macOS, where a ratio between two microbenchmarks is not measurable.
#[test]
fn inertialization_is_faster_on_the_clock() {
    const STEPS: usize = 900;
    const PERIOD: usize = 40;
    const ROUNDS: usize = 7;

    let mut best_inert = f64::INFINITY;
    let mut best_faded = f64::INFINITY;
    let mut sink = 0.0f32;
    for _ in 0..ROUNDS {
        // Interleaved, so a thermal or scheduling drift over the run cannot land
        // entirely on one of the two.
        let mut b = PoseBlender::new();
        let t = Instant::now();
        sink += run(&mut b, STEPS, PERIOD);
        best_inert = best_inert.min(t.elapsed().as_secs_f64());

        let mut b = PoseBlender::cross_fade();
        let t = Instant::now();
        sink += run(&mut b, STEPS, PERIOD);
        best_faded = best_faded.min(t.elapsed().as_secs_f64());
    }
    assert!(sink.is_finite());
    let ratio = best_faded / best_inert;
    println!(
        "inertialization {:.3} ms, cross-fade {:.3} ms over {STEPS} steps \
         (min of {ROUNDS}); cross-fade / inertialization = {ratio:.2}x",
        best_inert * 1e3,
        best_faded * 1e3
    );
    if cfg!(target_os = "macos") {
        // **macOS reports and does not assert, by name.** The runner is
        // paravirtualised and a ratio between two microbenchmarks is not a
        // measurement there. The functional twin above still asserts here.
        eprintln!("macOS: timing reported, not asserted (paravirtual runner)");
        return;
    }
    assert!(
        ratio > 1.0,
        "the cross-fade was not dearer on the clock: {:.3} ms against \
         inertialization's {:.3} ms",
        best_faded * 1e3,
        best_inert * 1e3
    );
}
