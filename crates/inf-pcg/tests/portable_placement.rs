//! **Nothing this crate commits is placed by libm** (Hardening Wave C, L6.F4).
//!
//! A grep, because the property is about *what the code is allowed to call* and
//! no numeric assertion in a single-process test can see it. The claim it
//! defends: a `.inf_pcg` document is evaluated independently by the editor and
//! by the shipped player, `phase19_gate`'s `bits()` folds every scattered
//! instance's rotation into a trace those two are compared on, `phase18_gate`
//! hashes the packed rotation block into a scatter content key, and
//! `PhysicsBridge3D` gives a `ScatteredSolid` a collider at the pose the
//! rotation names. The P14 law says `std` transcendentals are not bit-identical
//! across targets, so one of them anywhere on this path silently falsifies every
//! one of those comparisons the moment the two sides are two machines.
//!
//! # It was not hypothetical
//!
//! Until Wave C the scatter kernel built its random yaw with
//! `DQuat::from_rotation_y` and its slope alignment with
//! `DQuat::from_rotation_arc` — both of which reach `sin_cos` **inside glam**,
//! where no grep of this crate would ever have found them, and neither of which
//! any test in the repository could see. The crate already had the portable
//! spelling: `grammar::span`'s `euler_deg_to_quat` had been going through
//! `inf_math::psin64`/`pcos64` since P19.4, forty lines from the module that
//! did not.
//!
//! # Why this is an integration test and not a `#[cfg(test)] mod`
//!
//! The value arms that pin the portable rotations against glam's live in
//! `scatter.rs` itself and must *name* the banned constructors to compare
//! against them. A ban list in the same file would then be reading its own
//! reference implementations. Splitting the two puts the ban outside anything it
//! could accidentally certify.

/// Files whose **production half** must be free of platform-dependent
/// transcendentals, with the reason each is on the list.
///
/// Paths rather than `include_str!`, so the `\r\n` normalization below is
/// possible at all: `core.autocrlf = true` checks `.rs` out CRLF on Windows and
/// a newline-delimited read of an `include_str!` cannot fix that afterwards
/// (the P22 law; `.gitattributes` pins `*.rs text eol=lf` for exactly these
/// gates, and this belt goes with those braces).
const COMMITTED_PLACEMENT: [(&str, &str, &str); 2] = [
    (
        "scatter.rs",
        "src/scatter.rs",
        "every scattered instance's rotation rides `phase19_gate`'s trace and \
         `phase18_gate`'s content hash, and a `ScatteredSolid`'s collider is \
         placed at it",
    ),
    (
        "grammar/span.rs",
        "src/grammar/span.rs",
        "`axis_quat` and `euler_deg_to_quat` are the door every authored module \
         rotation in a committed `.inf_pcg` goes through, and `yaw_onto` orients \
         every span-placed module",
    ),
];

/// Every `std` transcendental that is not bit-portable across targets, both
/// spellings of each, plus the glam constructors that reach `sin_cos` **inside
/// another crate** where no grep of this one would see them.
///
/// `.sqrt()` is deliberately absent: IEEE-754 specifies it exactly, which is why
/// `yaw_onto` and `tilt_onto` are built out of it. So are `.abs()`, `.floor()`
/// and `.clamp()`.
const BANNED: [&str; 42] = [
    // Twelve method forms…
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
    // …each with BOTH width spellings of its UFCS twin, which is the only
    // version of this list that is a rule rather than a sample.
    "f32::sin(",
    "f64::sin(",
    "f32::cos(",
    "f64::cos(",
    "f32::tan(",
    "f64::tan(",
    "f32::asin(",
    "f64::asin(",
    "f32::acos(",
    "f64::acos(",
    "f32::atan(",
    "f64::atan(",
    "f32::atan2(",
    "f64::atan2(",
    "f32::sin_cos(",
    "f64::sin_cos(",
    "f32::cbrt(",
    "f64::cbrt(",
    "f32::powf(",
    "f64::powf(",
    "f32::exp(",
    "f64::exp(",
    "f32::ln(",
    "f64::ln(",
    // …plus the glam constructors, which reach `sin_cos` inside another crate.
    "from_rotation_arc",
    "from_axis_angle",
    "from_euler",
    "from_rotation_x(",
    "from_rotation_y(",
    "from_rotation_z(",
];

/// A file's production half with comment lines blanked.
///
/// Two exclusions, both load-bearing:
///
/// * **`#[cfg(test)]` and everything after it.** Both files carry exactly one
///   test module and it is the tail; the split is asserted below rather than
///   assumed, because a module inserted *earlier* would silently shrink what the
///   ban scans while leaving it green — this repository's recurring gate defect.
///   A test fixture is also allowed to name the banned constructors: it is the
///   thing that *checks* the portable spelling, and forcing `axis_quat` into it
///   would make the fixture agree with the code under test by construction.
/// * **comment lines**, because the doc comments on `axis_quat` and `tilt_onto`
///   name `from_axis_angle` and `from_rotation_arc` repeatedly *while explaining
///   why they are banned* — the P24.1 F1 finding, which is the sentence such a
///   gate is likeliest to be written next to.
fn production_code(relative: &str) -> String {
    let src =
        std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(relative))
            .unwrap_or_else(|e| panic!("{relative} is readable: {e}"))
            .replace("\r\n", "\n");
    let cut = src
        .find("#[cfg(test)]")
        .unwrap_or_else(|| panic!("{relative} has no test module; the split below is guessing"));
    assert_eq!(
        src.matches("#[cfg(test)]").count(),
        1,
        "{relative} has more than one `#[cfg(test)]`; cutting at the first would \
         discard production code after it, and this gate would then pass on less \
         while still passing"
    );
    src[..cut]
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

fn offenders(code: &str) -> Vec<&'static str> {
    BANNED
        .iter()
        .copied()
        .filter(|b| code.contains(b))
        .collect()
}

#[test]
fn committed_placement_uses_no_platform_dependent_trigonometry() {
    for (name, path, why) in COMMITTED_PLACEMENT {
        let code = production_code(path);
        let found = offenders(&code);
        assert!(
            found.is_empty(),
            "{name} calls {found:?} — {why}, and `std` transcendentals are not \
             bit-identical across targets (the P14 law). Use this crate's own \
             portable family: `axis_quat`, `euler_deg_to_quat`, `yaw_onto`, or \
             `inf_math`'s `psin64`/`pcos64`. `sqrt` is fine and is deliberately \
             not banned."
        );
    }
}

/// **NOT VACUOUS**: the files are on the list, they contain code, the portable
/// replacements are the ones actually used, and the ban rejects the call it was
/// written for.
#[test]
fn the_placement_ban_is_looking_at_real_code() {
    for (name, path, _) in COMMITTED_PLACEMENT {
        let code = production_code(path);
        assert!(
            code.lines().filter(|l| !l.trim().is_empty()).count() > 60,
            "{name} reduced to almost nothing after the split — the ban is \
             scanning an empty string"
        );
    }

    // The portable replacement really is used where the banned one was.
    let scatter = production_code(COMMITTED_PLACEMENT[0].1);
    assert!(
        scatter.contains("axis_quat(DVec3::Y, yaw)"),
        "the scatter yaw no longer goes through the portable axis quaternion"
    );
    assert!(
        scatter.contains("fn tilt_onto("),
        "the slope alignment no longer has its portable arc"
    );
    let span = production_code(COMMITTED_PLACEMENT[1].1);
    assert!(
        span.contains("psin64(h)") && span.contains("pcos64(h)"),
        "`axis_quat` no longer builds its half angle from the portable pair"
    );

    // **Built to falsify** (the P22 law): the same predicate over a poisoned copy
    // must reject it. The poison is the exact defect this gate was written for.
    let poisoned = format!("{scatter}\n    let q = tilt * DQuat::from_rotation_y(yaw);");
    assert!(
        !offenders(&poisoned).is_empty(),
        "the ban cannot see a `from_rotation_y` even when one is put in front of it"
    );
}

/// **Both spellings of every method form**, so the list is a rule rather than a
/// sample.
///
/// The P24.2 re-audit's minor 3, in the crate it had not reached: `.sin()` is a
/// substring ban and `f64::sin(x)` — the same call written the other way —
/// walks straight past it. Adding a method form without its UFCS twins now fails
/// here instead of failing silently in five years.
#[test]
fn the_ban_covers_both_spellings_of_the_functions_it_names() {
    let methods: Vec<&str> = BANNED
        .iter()
        .copied()
        .filter(|b| b.starts_with('.'))
        .collect();
    assert_eq!(
        methods.len(),
        12,
        "the method half of the list changed size; the counts below and the \
         UFCS half both depend on it"
    );
    for m in &methods {
        let bare = m.trim_start_matches('.').trim_end_matches(['(', ')']);
        for width in ["f32", "f64"] {
            let ufcs = format!("{width}::{bare}(");
            assert!(
                BANNED.contains(&ufcs.as_str()),
                "`{m}` is banned as a method but `{ufcs}` is not — a ban that \
                 enumerates one spelling is a sample, not a rule"
            );
        }
    }
    // Twelve methods × two widths, plus the six glam constructors.
    assert_eq!(BANNED.len(), 12 + 24 + 6);
}

/// **Round-2 finding R2.B**: this gate's list is a superset of the canonical
/// one, and passes the shared completeness meta-arm.
///
/// Six hand-copies of the libm ban existed and they had diverged — the erosion
/// mirror banned `f64::cos(` over an `f32` module, the fracture gate was
/// missing sixteen UFCS twins including `f64::cbrt(` and every glam entry, the
/// physics gate had no `.atan()`. None was a live escape, which is exactly the
/// problem with a list that enumerates what somebody thought of.
/// `inf_math::libm_ban::ALL` is the one list now; this arm is what keeps this
/// gate tied to it.
#[test]
fn this_gate_bans_everything_the_canonical_list_does() {
    let mine: Vec<&str> = BANNED.to_vec();
    inf_math::libm_ban::covers_both_spellings("inf-pcg/tests/portable_placement.rs", &mine);
    let missing: Vec<&str> = inf_math::libm_ban::ALL
        .iter()
        .copied()
        .filter(|b| !mine.contains(b))
        .collect();
    assert!(
        missing.is_empty(),
        "{} is missing {missing:?} from `inf_math::libm_ban::ALL` — six copies of this list diverged once already",
        "inf-pcg/tests/portable_placement.rs"
    );
}
