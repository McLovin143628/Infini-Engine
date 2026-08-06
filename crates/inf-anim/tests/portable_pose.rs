//! **The pose pipeline calls no `std` transcendental** (P24.2 audit M-SLERP).
//!
//! A grep, because the property is about *what the code is allowed to call* and
//! no numeric assertion in a single-process test can see it. The claim it
//! defends: an evaluated pose is folded into the sim's `state_bytes`, and
//! `state_bytes` is compared between the editor's Simulate and the shipped
//! player — and, through the replay and net paths, between two *machines*. The
//! P14 law says `std` `sin`/`cos`/`acos` are not bit-identical across targets, so
//! a single one of them on this path silently falsifies every determinism claim
//! the engine makes about animation.
//!
//! # It was not hypothetical
//!
//! The P24.2 audit found `Quat::slerp` — `acos_approx` plus three `sin`s — in
//! **both** of the two most-travelled blends in the crate:
//!
//! * `clip::QuatTrack::sample`, on every `Interpolation::Linear` track, which is
//!   the `#[default]` and what the glTF importer emits;
//! * `pose::blend_poses`, on every state-machine transition with a non-zero
//!   duration and in every blend space.
//!
//! Both are now `inf_math::pslerp`. This file is what stops them coming back —
//! and it covers `ik.rs`, whose module docs have claimed a trig-free property
//! since the day it was written with nothing enforcing it.
//!
//! # `sqrt` is deliberately absent from the ban
//!
//! IEEE-754 specifies it exactly, so it is bit-portable. That is why both IK
//! solvers are built out of it.

/// Files whose **whole item text** must be free of `std` transcendentals, with
/// the reason each one is on the list.
const SIM_PATH: [(&str, &str, &str); 4] = [
    (
        "pose.rs",
        include_str!("../src/pose.rs"),
        "blend_poses and skinning_matrices produce the pose that is folded into state_bytes",
    ),
    (
        "clip.rs",
        include_str!("../src/clip.rs"),
        "QuatTrack::sample is the default interpolation and the most-travelled blend in the engine",
    ),
    (
        "ik.rs",
        include_str!("../src/ik.rs"),
        "IK is a post-pass over that same pose, and its module docs have claimed this property since it was written",
    ),
    (
        "state_machine.rs",
        include_str!("../src/state_machine.rs"),
        "eval_pose chooses and cross-fades the poses the two above produce",
    ),
];

/// Every `std` transcendental that is not bit-portable across targets.
///
/// `.sqrt()` is **not** here, on purpose: IEEE-754 specifies it exactly. Neither
/// is `.abs()`, `.floor()` or `.clamp()`, for the same reason. `.cbrt()` IS here
/// — the P22.4 widening: on `wasm32` the standard library routes it through the
/// `libm` crate, so a browser client and a native one differ by an ulp.
const BANNED_CALLS: [&str; 12] = [
    ".sin()",
    ".cos()",
    ".tan()",
    ".asin()",
    ".acos()",
    ".atan()",
    ".atan2(",
    ".sin_cos()",
    ".cbrt()",
    ".powf(",
    ".exp()",
    ".ln()",
];

/// glam constructors that reach `sin_cos` **inside another crate**, where no grep
/// of this one would ever see them.
///
/// This half of the list is the one that has actually caught things: P23.5 found
/// `DQuat::from_axis_angle` doing it, `ik::rotation_between` exists because
/// `Quat::from_rotation_arc`'s antiparallel branch does it, and M-SLERP found
/// `Quat::slerp` doing it twice.
const BANNED_GLAM: [&str; 8] = [
    ".slerp(",
    "from_rotation_arc",
    "from_axis_angle",
    "from_euler",
    "from_rotation_x(",
    "from_rotation_y(",
    "from_rotation_z(",
    "to_euler(",
];

/// Source with every `//` comment line blanked, and the `#[cfg(test)]` tail cut
/// off.
///
/// Two exclusions, both load-bearing:
///
/// * **comments** — the module docs of these files name `sin`, `slerp` and
///   `from_rotation_arc` repeatedly *while explaining why they are banned*, which
///   is the sentence such a gate is most likely to be written next to (the P24.1
///   F1 finding, in its general form);
/// * **`#[cfg(test)]`** — a fixture may build a rotation from an angle. It is not
///   on the sim path; it is the thing that *checks* the sim path, and forcing
///   `pslerp` into a test fixture would make the fixture agree with the code
///   under test by construction.
fn production_code(src: &str) -> String {
    let src = src.replace("\r\n", "\n");
    let src = match src.find("\n#[cfg(test)]\n") {
        Some(i) => src[..i].to_string(),
        None => src,
    };
    src.lines()
        .map(|l| {
            if l.trim_start().starts_with("//") {
                ""
            } else {
                l
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn the_animation_blend_uses_no_platform_dependent_trigonometry() {
    for (name, src, why) in SIM_PATH {
        let code = production_code(src);
        for banned in BANNED_CALLS.iter().chain(BANNED_GLAM.iter()) {
            let hits: Vec<usize> = code
                .lines()
                .enumerate()
                .filter(|(_, l)| l.contains(banned))
                .map(|(i, _)| i + 1)
                .collect();
            assert!(
                hits.is_empty(),
                "{name} calls `{banned}` at line(s) {hits:?} — {why}, and `std` \
                 transcendentals are not bit-identical across targets (the P14 \
                 law). Use `inf_math`'s portable family: `pslerp`, `psin64`, \
                 `pcos64`, `pacos64`. `sqrt` is fine and is deliberately not \
                 banned."
            );
        }
    }
}

/// **NOT VACUOUS**: the files really are on the list, they really do contain
/// code, and the ban really does catch the thing it was written for.
#[test]
fn the_trig_ban_is_looking_at_real_code() {
    for (name, src, _) in SIM_PATH {
        let code = production_code(src);
        assert!(
            code.lines().filter(|l| !l.trim().is_empty()).count() > 40,
            "{name} reduced to nothing after stripping comments and tests — the \
             ban above is scanning an empty string"
        );
        // The pose path is reached, not merely present.
        assert!(
            code.contains("pub fn") || code.contains("pub(crate) fn"),
            "{name} exposes nothing"
        );
    }
    // The portable replacement really is used where the banned one was.
    let pose = production_code(SIM_PATH[0].1);
    let clip = production_code(SIM_PATH[1].1);
    assert!(
        pose.contains("inf_math::pslerp"),
        "pose.rs no longer blends through the portable arc"
    );
    assert!(
        clip.contains("inf_math::pslerp"),
        "clip.rs no longer blends through the portable arc"
    );
    // And the ban is a real filter: it fires on a string that contains one.
    let decoy = "let q = a.slerp(b, 0.5);";
    assert!(
        BANNED_GLAM.iter().any(|b| decoy.contains(b)),
        "the ban list would not catch a literal `slerp` call"
    );
}
