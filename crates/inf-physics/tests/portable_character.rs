//! **The character step's math is portable** (P29.4 audit, A6) — a source gate
//! over the three `d3` modules that write a character's `Transform`.
//!
//! # Why these three files, and why now
//!
//! `inf-anim`'s `portable_pose` gate covers the modules that produce a *pose*,
//! and it grew three entries this wave (`warp.rs`, `foot.rs`, `ragdoll.rs`).
//! What it does not cover — because it is a different crate, and its list is
//! `include_str!`ed from `inf-anim/src` plus the two hosts — is the code that
//! takes those answers and puts a character somewhere:
//!
//! * `d3/movement.rs` — the fixed step itself. It writes `Transform.translation`
//!   and `Transform.rotation.y` for every character in the world, and since
//!   P29.4 that includes the mantle's warped placement.
//! * `d3/traversal.rs` — the ledge probe and the land-prediction sweep. The
//!   probe's answer *is* where a mantle ends; the sweep's answer feeds the
//!   landing classifier, which chooses between a soft landing, a hard one, a
//!   roll and a ragdoll.
//! * `d3/ragdoll_bridge.rs` — reads the pelvis's rotation to decide which way up
//!   a character is and where its capsule goes.
//!
//! Every one of those values is folded into `state_bytes` and compared, byte for
//! byte, by `inf-player`'s `movement_parity` — between two hosts today and
//! between two machines by the claim the gate makes. The P14 law says a value
//! two machines re-derive independently may not depend on a libm, and until this
//! file existed nothing enforced it on this side of the seam.
//!
//! The three files are clean today. That is the point: this gate is what keeps
//! the next `.atan2(` out, and the repository's own experience is that a rule
//! with no gate holds until exactly the moment it matters.

/// The three modules, with the reason each one is on the list.
const CHARACTER_PATH: [(&str, &str, &str); 3] = [
    (
        "d3/movement.rs",
        include_str!("../src/d3/movement.rs"),
        "the fixed step writes every character's Transform, including the mantle's warped placement, and both hosts run it",
    ),
    (
        "d3/traversal.rs",
        include_str!("../src/d3/traversal.rs"),
        "the ledge probe decides where a mantle ENDS and the land-prediction sweep decides which landing the classifier reaches",
    ),
    (
        "d3/ragdoll_bridge.rs",
        include_str!("../src/d3/ragdoll_bridge.rs"),
        "the pelvis read chooses supine versus prone and places the capsule, and both are in the parity trace",
    ),
];

/// A marker that must be present in each file, so the gate cannot pass because
/// it is scanning something that is no longer the module it names.
const ANCHORS: [(&str, &[&str]); 3] = [
    (
        "d3/movement.rs",
        &["inf_anim::warp_offset(", "inf_math::pacos64("],
    ),
    (
        "d3/traversal.rs",
        &["inf_math::pacos64(", "inf_math::patan2_64("],
    ),
    (
        "d3/ragdoll_bridge.rs",
        &["inf_math::proll(", "inf_math::pyaw("],
    ),
];

/// Comment lines blanked, CRLF normalized — the `fracture_3d` recipe, for the
/// same two reasons: these files *name* the banned spellings while explaining
/// why they are banned (the P24.1 F1 finding), and `core.autocrlf = true` checks
/// `.rs` out CRLF on Windows (the P22 law).
fn production_code(src: &str) -> String {
    src.replace("\r\n", "\n")
        .lines()
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
fn the_character_step_calls_no_libm() {
    const GATE: &str = "inf-physics/tests/portable_character.rs";
    /// Beyond the canonical list: both reach `sin_cos` inside glam, and neither
    /// is a method call the substring bans would see.
    const EXTRA: [&str; 4] = [
        "to_euler(",
        ".slerp(",
        "from_rotation_y(",
        "from_rotation_x(",
    ];
    let banned: Vec<&str> = inf_math::libm_ban::ALL
        .iter()
        .copied()
        .chain(EXTRA)
        .collect();
    // One list, not six (round-2 finding R2.B): this copy is checked for
    // completeness rather than trusted.
    inf_math::libm_ban::covers_both_spellings(GATE, &banned);

    let offenders = |text: &str| -> Vec<(&'static str, Vec<usize>)> {
        banned
            .iter()
            .filter_map(|b| {
                let hits: Vec<usize> = text
                    .lines()
                    .enumerate()
                    .filter(|(_, l)| l.contains(b))
                    .map(|(i, _)| i + 1)
                    .collect();
                (!hits.is_empty()).then_some((*b, hits))
            })
            .collect()
    };

    for (name, src, why) in CHARACTER_PATH {
        // No `#[cfg(test)]` region, so the scope is the whole file. Asserted
        // rather than assumed: the day one grows a test module, a fixture that
        // builds a rotation from an angle would fail this gate for the wrong
        // reason, and the fix is to strip the region the way `inf-anim`'s
        // `portable_pose::production_code` does.
        assert!(
            !src.contains("#[cfg(test)]"),
            "`{name}` grew a test module; this gate scans the whole file and must \
             learn to strip `#[cfg(test)]` regions first"
        );
        let code = production_code(src);
        assert!(
            code.lines().filter(|l| !l.trim().is_empty()).count() > 100,
            "`{name}` reduced to nothing after blanking comments — the ban is \
             scanning an empty string"
        );
        let anchors = ANCHORS
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, a)| *a)
            .expect("every file on the list has anchors");
        for anchor in anchors {
            assert!(
                code.contains(anchor),
                "`{name}` no longer contains `{anchor}` — this gate is no longer \
                 covering the code it was written for"
            );
        }
        let found = offenders(&code);
        assert!(
            found.is_empty(),
            "`{name}` calls libm: {found:?} — it is not bit-portable (the P14 \
             law), and {why}. Use `inf_math`'s portable family (`psin64`, \
             `pcos64`, `pacos64`, `patan2_64`, `pyaw`, `proll`, `pslerp`); \
             `sqrt` is fine and is deliberately not banned."
        );

        // **Built to falsify.** The same predicate over a poisoned copy must
        // reject it, or the assertion above is a statement about a string that
        // happens to be green. The poison is the shape that would actually be
        // written here: an angle taken off a quaternion the easy way.
        let poisoned = format!("{code}\n    let roll = q.to_euler(EulerRot::YXZ).2;");
        assert!(
            !offenders(&poisoned).is_empty(),
            "the ban cannot see a `to_euler` in `{name}` even when one is put in \
             front of it"
        );
        let poisoned = format!("{code}\n    let a = (y as f64).atan2(x as f64);");
        assert!(
            !offenders(&poisoned).is_empty(),
            "the ban cannot see an `.atan2(` in `{name}` even when one is put in \
             front of it"
        );
    }
}
