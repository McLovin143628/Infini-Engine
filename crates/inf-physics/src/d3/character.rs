//! Kinematic character mover (3D) — the `d3` mirror of [`crate::d2`]'s mover and
//! the substrate a 3D Blueprint character-controller kit builds on (P12).

use glam::DVec3;
use rapier3d_f64::control::{CharacterAutostep, CharacterLength, KinematicCharacterController};
use rapier3d_f64::prelude::{Pose, QueryPipeline, SharedShape};

use super::world::ColliderShape3D;
use super::ColliderId3D;

/// The outcome of a [`move_character`](crate::d3::PhysicsWorld3D::move_character)
/// call: how far the character actually moved, whether it ended on the ground,
/// and what it touched.
#[derive(Clone, Debug, PartialEq)]
pub struct CharacterMove3D {
    /// The translation to apply to the character (already collision-resolved and
    /// slid). Add this to the character's position.
    pub translation: DVec3,
    /// Whether the character is touching the ground after the move.
    pub grounded: bool,
    /// Colliders the character brushed against during the move, sorted
    /// deterministically and de-duplicated.
    pub collisions: Vec<ColliderId3D>,
}

/// **Autostep configuration** (P29.3) — the parameters that make stairs work,
/// in absolute metres.
///
/// rapier's own defaults are *relative* to the swept shape, which is wrong for a
/// character that changes height: a crouched capsule would step over less than a
/// standing one, so a doorway sill that is climbable upright would stop a crouch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutoStep3D {
    /// The tallest obstacle the character steps onto, world metres.
    pub max_height: f64,
    /// Free space that must exist beyond the step for it to be taken, metres.
    /// Without it a character steps onto a ledge it cannot stand on.
    pub min_width: f64,
    /// Whether the character steps onto dynamic bodies as well as static
    /// geometry. rapier defaults this on and so does the movement component:
    /// a plank on the floor is a step.
    pub include_dynamic_bodies: bool,
}

/// A `move_and_slide`-style kinematic character mover.
///
/// It wraps rapier's `KinematicCharacterController` plus the character's collision
/// shape. Construct one, tune it with the fluent setters, and drive it each fixed
/// step with [`PhysicsWorld3D::move_character`](crate::d3::PhysicsWorld3D::move_character).
/// The mover holds no position of its own — the caller owns the character's
/// transform — which keeps it a pure primitive the Blueprint kit can compose.
#[derive(Clone)]
pub struct CharacterMover3D {
    controller: KinematicCharacterController,
    shape: SharedShape,
}

impl CharacterMover3D {
    /// A mover for the given character shape, with sensible defaults: up is `+Y`,
    /// sliding on, a 45° max climbable slope, ground-snap on, and autostep off
    /// (opt in with [`autostep`](Self::autostep) — see that method for why the
    /// default being off was, until P29.3, the reason stairs did not work).
    ///
    /// # Panics
    /// Panics if `shape` is a degenerate trimesh (a mover shape should always be a
    /// simple primitive — sphere/box/capsule — so this only trips on misuse).
    pub fn new(shape: ColliderShape3D) -> Self {
        Self {
            controller: KinematicCharacterController::default(),
            shape: shape
                .to_shared()
                .expect("character mover shape must be a valid (non-trimesh) shape"),
        }
    }

    /// Set which way is "up" (used to decide what counts as floor). Default `+Y`.
    pub fn up(mut self, up: DVec3) -> Self {
        self.controller.up = up;
        self
    }

    /// The small gap kept between the character and the world (absolute world
    /// units). Must be non-zero for numerical stability.
    pub fn offset(mut self, offset: f64) -> Self {
        self.controller.offset = CharacterLength::Absolute(offset);
        self
    }

    /// Whether the character slides along surfaces it hits (vs. stopping dead).
    pub fn slide(mut self, slide: bool) -> Self {
        self.controller.slide = slide;
        self
    }

    /// The maximum slope angle (radians, measured from `up`) the character can
    /// walk up.
    pub fn max_slope_climb_angle(mut self, radians: f64) -> Self {
        self.controller.max_slope_climb_angle = radians;
        self
    }

    /// The minimum slope angle (radians) at which the character starts sliding
    /// back down.
    pub fn min_slope_slide_angle(mut self, radians: f64) -> Self {
        self.controller.min_slope_slide_angle = radians;
        self
    }

    /// Snap the character to the ground when within this absolute distance
    /// (`None` disables snapping — the character will fly off ledges/ramps).
    pub fn snap_to_ground(mut self, distance: Option<f64>) -> Self {
        self.controller.snap_to_ground = distance.map(CharacterLength::Absolute);
        self
    }

    /// **Automatic stepping over small obstacles — stairs** (P29.3). `None`
    /// disables it.
    ///
    /// # Why this signature changed
    ///
    /// It used to be `autostep(bool)`, using rapier's default
    /// (`Relative(0.25)` of the shape's height, `Relative(0.5)` width). It had
    /// **no production caller at all**: `.autostep(` appeared exactly twice in
    /// the tree, at its own definition here and in the `d2` sibling. So rapier's
    /// real default — `None` — applied to every character this engine has ever
    /// simulated, and stairs did not work. That is not a subtle failure: a
    /// character walks into a 20 cm step and stops.
    ///
    /// A bool cannot express what a character needs, because a step height is a
    /// *tuning* value that belongs beside the slope limit and the capsule
    /// heights, and because `Relative` measures against the shape — so a
    /// crouched capsule would silently step over less. Absolute metres, from
    /// [`CharacterMovement::step_height_m`](../../inf_ecs/components/struct.CharacterMovement.html),
    /// keep the stairs a character can climb independent of what it is doing.
    pub fn autostep(mut self, step: Option<AutoStep3D>) -> Self {
        self.controller.autostep = step.map(|s| CharacterAutostep {
            max_height: CharacterLength::Absolute(s.max_height.max(0.0)),
            min_width: CharacterLength::Absolute(s.min_width.max(0.0)),
            include_dynamic_bodies: s.include_dynamic_bodies,
        });
        self
    }

    /// Replace the swept shape, keeping every tuning value (P29.3).
    ///
    /// The movement step decides a stance change and must sweep with the capsule
    /// it just decided, one step before the component that holds it is written.
    /// Rebuilding the whole mover to change one field would duplicate
    /// `mover_for`'s reasoning at the call site, which is the thing that pair
    /// was retired for.
    pub fn with_shape(mut self, shape: ColliderShape3D) -> Self {
        if let Some(s) = shape.to_shared() {
            self.shape = s;
        }
        self
    }

    /// Run the controller against a prepared query pipeline. Facade-internal —
    /// callers use `PhysicsWorld3D::move_character`.
    pub(crate) fn solve(
        &self,
        pipe: &QueryPipeline,
        dt: f64,
        position: DVec3,
        desired_translation: DVec3,
    ) -> CharacterMove3D {
        let character_pos = Pose::from_translation(position);
        let mut collisions: Vec<ColliderId3D> = Vec::new();
        let movement = self.controller.move_shape(
            dt,
            pipe,
            &*self.shape,
            &character_pos,
            desired_translation,
            |collision| collisions.push(ColliderId3D(collision.handle)),
        );
        collisions.sort_unstable();
        collisions.dedup();
        CharacterMove3D {
            translation: movement.translation,
            grounded: movement.grounded,
            collisions,
        }
    }
}
