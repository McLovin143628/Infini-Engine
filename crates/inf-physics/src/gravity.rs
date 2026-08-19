//! **The two solvers' gravity, as one value** (P29.7).
//!
//! # The defect this type exists to close
//!
//! `LevelSettings` has carried a `gravity_2d` *and* a `gravity_3d` since the
//! scene schema had a version number, and until this wave **only the first one
//! was ever read**. Both hosts built their 3D world with
//! `PhysicsBridge3D::new(DVec3::new(0.0, gravity_2d.y, 0.0))`, so a level's
//! `gravity_3d` was authored in the Details panel, serialized into the `.inf_lvl`,
//! round-tripped by the codec, carried through every schema ladder — and read by
//! nothing that simulates.
//!
//! The P29.6 audit found it and, writing the cook advisory for it, found the
//! bigger half: [`LevelSettings::default()`] pairs a `gravity_2d` of **zero**
//! with a `gravity_3d` of −9.81, so **every level that never touched either
//! field had no 3D gravity at all** while its Details panel read −9.81. Two
//! committed samples were in that state. Nothing had noticed because a
//! *character* carries its own `CharacterMovement::gravity_mps2` and never asks
//! the world for one — only a **dynamic body** reads this number, and until this
//! wave the samples with dynamic bodies were exactly the samples whose authors
//! had typed `gravity_2d.y` by hand to make them fall.
//!
//! # The decision
//!
//! **Each field feeds the solver its name says.** The 2D world is built from
//! `gravity_2d` and the 3D world from `gravity_3d`; neither default moves.
//!
//! The alternative was to change `LevelSettings::default()`'s `gravity_2d` to
//! −9.81 so the existing wiring became correct. It is the worse half: a 2D
//! project's default would stop being "the game decides", every 2D level in the
//! tree would gain a downward pull it never asked for, and `gravity_3d` would
//! still be a field nobody reads. The pairing is not a bug once each field feeds
//! its own dimension — a 2D game choosing its own gravity and a 3D level standing
//! on Earth are both right defaults, and they are only incoherent while one field
//! is doing both jobs.
//!
//! **What moved in the tree.** Every committed 3D sample that had typed
//! `gravity_2d.y = -9.81` also carries `gravity_3d = (0, -9.81, 0)` (four of them
//! explicitly, one by the default), so its 3D gravity is the number it always
//! was and **not one byte of committed content changed**. The levels that change
//! behaviour are exactly the ones the audit named: the levels that never set
//! either field and therefore had none. They have it now, which is the fix.
//!
//! **What the cook says now.** The advisory inverts. `ignored_gravity_3d` said
//! "your `gravity_3d` is read by nothing"; that sentence is false as of this
//! wave. Its replacement, [`crate::gravity`]'s sibling in `inf_packager::cook`,
//! fires on the one silent hazard the change creates: a level with 3D content
//! whose `gravity_2d.y` used to be its 3D gravity and whose `gravity_3d` says
//! something else.

use glam::{DVec2, DVec3};

/// The gravity **both** solvers are built with — the 2D world's and the 3D
/// world's, carried together because a level authors them together.
///
/// See the [module docs](self) for why this is a pair rather than one vector and
/// for the P29.7 decision it records.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldGravity {
    /// The 2D solver's gravity, m/s² — `LevelSettings::gravity_2d`.
    pub d2: DVec2,
    /// The 3D solver's gravity, m/s² — `LevelSettings::gravity_3d`.
    pub d3: DVec3,
}

impl WorldGravity {
    /// Earth, in both dimensions: `(0, −9.81)` and `(0, −9.81, 0)`.
    ///
    /// The value a fixture means when it says "gravity"; identical to
    /// [`from_2d`](Self::from_2d) of `(0, −9.81)`, and identical to what
    /// `LevelSettings::default()`'s `gravity_3d` gives a level that never touched
    /// the field.
    pub const EARTH: Self = Self {
        d2: DVec2::new(0.0, -9.81),
        d3: DVec3::new(0.0, -9.81, 0.0),
    };

    /// No gravity in either dimension.
    pub const NONE: Self = Self {
        d2: DVec2::ZERO,
        d3: DVec3::ZERO,
    };

    /// The pair a level authors: the two fields, each feeding its own solver.
    pub const fn new(d2: DVec2, d3: DVec3) -> Self {
        Self { d2, d3 }
    }

    /// **The pre-P29.7 pairing**: one 2D vector whose vertical component drives
    /// *both* solvers.
    ///
    /// This is what `RuntimeSim::new` and `SimSession::enter` did unconditionally
    /// before this wave, and it is kept — with this name and this doc — because a
    /// fixture that builds a world by hand and passes one number means that
    /// number in both dimensions, and rewriting a hundred of them to say so twice
    /// would be churn rather than clarity.
    ///
    /// **A host must not call it.** A host has a level, a level has two authored
    /// fields, and collapsing them is the defect this type closes;
    /// `both_hosts_build_the_3d_solver_from_the_authored_gravity_3d` is the pin
    /// that says so.
    pub const fn from_2d(d2: DVec2) -> Self {
        Self {
            d2,
            d3: DVec3::new(0.0, d2.y, 0.0),
        }
    }
}

impl Default for WorldGravity {
    /// [`EARTH`](Self::EARTH) — the same default `LevelSettings::gravity_3d`
    /// carries, so a `WorldGravity::default()` and an untouched level agree.
    fn default() -> Self {
        Self::EARTH
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_legacy_pairing_puts_the_2d_vertical_on_the_3d_solver() {
        let g = WorldGravity::from_2d(DVec2::new(0.0, -20.0));
        assert_eq!(g.d2, DVec2::new(0.0, -20.0));
        assert_eq!(g.d3, DVec3::new(0.0, -20.0, 0.0));
    }

    #[test]
    fn earth_is_the_legacy_pairing_of_earth() {
        assert_eq!(WorldGravity::EARTH, WorldGravity::from_2d(DVec2::new(0.0, -9.81)));
        assert_eq!(WorldGravity::default(), WorldGravity::EARTH);
    }

    #[test]
    fn the_authored_pair_keeps_both_numbers() {
        let g = WorldGravity::new(DVec2::ZERO, DVec3::new(0.0, -1.62, 0.0));
        assert_eq!(g.d2, DVec2::ZERO, "a 2D game with no gravity keeps none");
        assert_eq!(
            g.d3.y, -1.62,
            "and its 3D solver reads the field with its own name"
        );
    }
}
