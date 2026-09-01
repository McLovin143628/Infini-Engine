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
//! **And since island wave VEH2a, the two VEHICLE modules** —
//! `inf_ecs::vehicle` and `d3/vehicle.rs`. That was a real hole, not a
//! tidy-up: the vehicle model writes a chassis's `Transform` and every wheel's,
//! its pose is compared byte for byte between two hosts by `island_gate`'s
//! drive arm, and the *only* thing keeping `f64::sin` out of it was a comment.
//! Every libm gate in the tree was checked when the hole was found —
//! `portable_character`'s own list (four `d3` files, neither of them the
//! vehicle), `portable_pose`'s thirty-five `include_str!`s (`inf-ecs`'s pose,
//! camera, movement, crowd and society, and not its vehicle) and
//! `inf-editor-core`'s crate-wide walk (a different crate) — and **not one of
//! them covered either file**. A cross-target divergence in a car would have
//! shipped: the two arms that compare a drive both run two hosts on one machine,
//! which is exactly the comparison a libm difference is invisible to.
//!
//! The files are clean today. That is the point: this gate is what keeps the
//! next `.atan2(` out, and the repository's own experience is that a rule with
//! no gate holds until exactly the moment it matters.

/// The modules, with the reason each one is on the list.
/// The fourth element is the **minimum** number of non-comment lines the file
/// must still have. It is per file rather than one number for all of them
/// because `d3/camera.rs` is deliberately the smallest door on this list — its
/// model lives in `inf_ecs::camera` and what is here is the world half — and a
/// single floor would either be vacuous for the big three or unreachable for it.
/// The guard exists so the ban cannot pass by scanning a file that has been
/// emptied out from under it.
const CHARACTER_PATH: [(&str, &str, &str, usize); 6] = [
    (
        "d3/camera.rs",
        include_str!("../src/d3/camera.rs"),
        "the camera is not sim state, and it is here because `phase29_gate` asserts a DETERMINISTIC camera trace: a claim that only holds on one target is a claim about this machine",
        50,
    ),
    (
        "d3/movement.rs",
        include_str!("../src/d3/movement.rs"),
        "the fixed step writes every character's Transform, including the mantle's warped placement, and both hosts run it",
        100,
    ),
    (
        "d3/traversal.rs",
        include_str!("../src/d3/traversal.rs"),
        "the ledge probe decides where a mantle ENDS and the land-prediction sweep decides which landing the classifier reaches",
        100,
    ),
    (
        "d3/ragdoll_bridge.rs",
        include_str!("../src/d3/ragdoll_bridge.rs"),
        "the pelvis read chooses supine versus prone and places the capsule, and both are in the parity trace",
        100,
    ),
    (
        "d3/vehicle.rs",
        include_str!("../src/d3/vehicle.rs"),
        "the vehicle door casts the wheel rays and writes every wheel entity's Transform, and `island_gate` compares a drive byte for byte between two hosts",
        100,
    ),
    (
        "inf-ecs/src/vehicle.rs",
        include_str!("../../inf-ecs/src/vehicle.rs"),
        "the whole driving model — the tyre, the torque curve, the gearbox and the steering — and its answers reach a chassis pose that both hosts compare",
        400,
    ),
];

/// A marker that must be present in each file, so the gate cannot pass because
/// it is scanning something that is no longer the module it names.
const ANCHORS: [(&str, &[&str]); 6] = [
    ("d3/camera.rs", &["inf_ecs::camera::", "cast_shape_where("]),
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
    (
        "d3/vehicle.rs",
        &["cast_ray_where(", "apply_force_at_point("],
    ),
    (
        "inf-ecs/src/vehicle.rs",
        &["inf_math::pcos64(", "inf_math::psin64(", "curve_bias("],
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

    for (name, src, why, min_lines) in CHARACTER_PATH {
        // **PRODUCTION ONLY.** The first four files on this list had no test
        // module and the gate asserted so, with a comment saying that the day one
        // grew a test module the fix was to strip the region the way `inf-anim`'s
        // `portable_pose::production_code` does. Wave VEH2a put
        // `inf-ecs/src/vehicle.rs` on the list, which has a large one — so that
        // is what happens now. The cut is at a COLUMN-ZERO `#[cfg(test)]`, which
        // is where a module-level test block begins and where an inner attribute
        // cannot be; `min_lines` below is what stops an over-eager cut from
        // leaving the ban scanning nothing.
        let whole = src.replace("\r\n", "\n");
        let src = match whole.find("\n#[cfg(test)]") {
            Some(i) => &whole[..i],
            None => &whole[..],
        };
        let code = production_code(src);
        assert!(
            code.lines().filter(|l| !l.trim().is_empty()).count() >= min_lines,
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
