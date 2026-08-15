//! **The libm ban list, once** — the canonical enumeration of the spellings a
//! source gate must refuse (round-2 finding **R2.B**).
//!
//! # The law
//!
//! `f32`/`f64` transcendental methods are **not bit-portable**. `sin`, `cos`,
//! `tan`, their inverses, `cbrt`, `powf`, `exp` and `ln` route through the
//! platform's libm on native targets and through the `libm` *crate* on wasm32,
//! and those are different implementations that are entitled to disagree in the
//! last ulp. The P14 law that came out of that is not "avoid them" — it is that
//! **a value two machines re-derive independently may not depend on one**, which
//! covers every committed byte, every cook, every derived asset and every trace a
//! PIE-==-shipping gate compares. [`crate::portable`] is the replacement set
//! (`psin`, `pcos`, `pcbrt`, `patan2_64`, `pslerp`, …), and the way the law is
//! *enforced* is a source gate per module that must obey it.
//!
//! # Why this list lives here rather than in each gate
//!
//! Because it did live in each gate, and they diverged. Round 2 found **six
//! hand-copies** with three named holes:
//!
//! * the erosion mirror banned `f64::cos(` over a module whose parameters are
//!   `f32`, so `f32::cos(` walked past it;
//! * the fracture gate was missing **sixteen** UFCS twins, including
//!   `f64::cbrt(` — the exact function the P22 law was extended for — and all
//!   five glam entries;
//! * the physics gate had no `.atan()` at all.
//!
//! None of them was a live escape (round 2 grepped), which is precisely the
//! problem: a ban list that enumerates what somebody thought of passes for years
//! and then does not catch the one that matters. One list, imported by every
//! gate, plus [`covers_both_spellings`] as a per-gate completeness meta-arm —
//! the `inf-pcg` pattern, hoisted.
//!
//! # What is deliberately NOT banned
//!
//! `sqrt`, `abs`, `floor`, `ceil`, `round`, `trunc`, `copysign`, `mul_add` and
//! the arithmetic operators. IEEE-754 specifies all of them **exactly**, which
//! is why the portable replacements are built out of them (`yaw_onto` and
//! `tilt_onto` in `inf-pcg`, the rejection samplers in `inf-render::debris`).
//! Banning them would ban the fix.

/// The twelve method spellings, e.g. `".sin()"`.
///
/// `atan2`, `sin_cos` and `powf` take arguments, so they end at the open paren;
/// the rest are closed, because `.sin()` must not also match `.sinh()`.
pub const METHODS: [&str; 12] = [
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

/// Both width spellings of every [`METHODS`] entry's UFCS twin, e.g.
/// `"f64::sin("`.
///
/// This half is the P24.2 re-audit's minor 3: `.sin()` is a *substring* ban and
/// `f64::sin(x)` — the same call written the other way — walks straight past it.
/// A list with only the method forms is a sample, not a rule.
pub const UFCS: [&str; 24] = [
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
];

/// The glam constructors that reach `sin_cos` **inside another crate**, where no
/// grep of the calling one would ever see them.
///
/// This is the half a hand-copy loses first, because nothing about the spelling
/// says "transcendental".
pub const GLAM: [&str; 6] = [
    "from_rotation_arc",
    "from_axis_angle",
    "from_euler",
    "from_rotation_x(",
    "from_rotation_y(",
    "from_rotation_z(",
];

/// Every banned spelling: [`METHODS`] + [`UFCS`] + [`GLAM`], 42 in all.
pub const ALL: [&str; 42] = {
    let mut out = [""; 42];
    let mut i = 0;
    while i < METHODS.len() {
        out[i] = METHODS[i];
        i += 1;
    }
    let mut j = 0;
    while j < UFCS.len() {
        out[METHODS.len() + j] = UFCS[j];
        j += 1;
    }
    let mut k = 0;
    while k < GLAM.len() {
        out[METHODS.len() + UFCS.len() + k] = GLAM[k];
        k += 1;
    }
    out
};

/// **The completeness meta-arm** every gate calls on its own list.
///
/// Panics naming the hole when `list` bans a canonical method spelling without
/// both of its UFCS twins, or omits any of [`GLAM`].
///
/// A gate may **narrow** the method half — some legitimately do, e.g. a module
/// allowed `powf` under a recorded bound — and may **add** extras of its own;
/// only the canonical methods it does carry are checked for twins. The glam
/// half is required outright, because it is the half a hand-copy loses first
/// (nothing about `from_euler` says "transcendental") and it is the half that
/// has actually caught things.
///
/// `who` is the gate's own name, so a failure says which file to open.
pub fn covers_both_spellings(who: &str, list: &[&str]) {
    // Only the CANONICAL method spellings are checked for UFCS twins. A gate
    // may add extras of its own — `.slerp(` and `to_euler(` are glam methods
    // two of them ban — and `f32::slerp(` is not a thing, so demanding a twin
    // for every dotted entry would make the meta-arm reject the very lists it
    // exists to complete. (Measured: it did, on the physics gate, on its first
    // run.)
    for m in list.iter().filter(|b| METHODS.contains(b)) {
        let bare = m.trim_start_matches('.').trim_end_matches(['(', ')']);
        for width in ["f32", "f64"] {
            let ufcs = format!("{width}::{bare}(");
            assert!(
                list.contains(&ufcs.as_str()),
                "{who}: `{m}` is banned as a method but `{ufcs}` is not — a ban \
                 that enumerates one spelling is a sample, not a rule. The \
                 canonical list is `inf_math::libm_ban::ALL`."
            );
        }
    }
    for g in GLAM {
        assert!(
            list.contains(&g),
            "{who}: `{g}` is missing — it reaches `sin_cos` inside glam, where no \
             grep of the calling crate would ever see it. The canonical list is \
             `inf_math::libm_ban::ALL`."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The canonical list is itself complete — the meta-arm applied to the
    /// thing every gate imports.
    #[test]
    fn the_canonical_list_passes_its_own_completeness_check() {
        covers_both_spellings("inf_math::libm_ban::ALL", &ALL);
        assert_eq!(ALL.len(), METHODS.len() + UFCS.len() + GLAM.len());
        assert_eq!(ALL.len(), 42);
        // No duplicates, or a count is not a count.
        let unique: std::collections::BTreeSet<&str> = ALL.iter().copied().collect();
        assert_eq!(unique.len(), ALL.len(), "the list repeats a spelling");
        // …and none of the exactly-specified operations crept in.
        for exact in [
            ".sqrt()",
            ".abs()",
            ".floor()",
            ".ceil()",
            ".round()",
            ".trunc()",
            ".mul_add(",
        ] {
            assert!(
                !ALL.contains(&exact),
                "{exact} is IEEE-exact and the portable replacements are built out \
                 of it; banning it bans the fix"
            );
        }
    }

    /// The meta-arm can see a hole — a gate that cannot fail is not a gate.
    #[test]
    fn the_meta_arm_sees_a_missing_twin_and_a_missing_constructor() {
        let missing_twin: Vec<&str> = ALL.iter().copied().filter(|b| *b != "f32::cos(").collect();
        assert!(
            std::panic::catch_unwind(|| covers_both_spellings("decoy", &missing_twin)).is_err()
        );

        let missing_glam: Vec<&str> = ALL
            .iter()
            .copied()
            .filter(|b| *b != "from_rotation_y(")
            .collect();
        assert!(
            std::panic::catch_unwind(|| covers_both_spellings("decoy", &missing_glam)).is_err()
        );

        // …and a legitimately NARROWED list still passes, so the meta-arm bounds
        // completeness rather than forcing every gate to ban everything.
        let narrowed: Vec<&str> = METHODS
            .iter()
            .copied()
            .filter(|m| *m == ".sin()" || *m == ".cos()")
            .chain(
                UFCS.iter()
                    .copied()
                    .filter(|u| u.ends_with("::sin(") || u.ends_with("::cos(")),
            )
            .chain(GLAM)
            .collect();
        covers_both_spellings("narrowed", &narrowed);
    }
}
