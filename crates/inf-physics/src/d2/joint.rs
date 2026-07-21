//! Joints (P12.1): constraints linking two bodies in a [`PhysicsWorld2D`]. The
//! `d2` sibling of [`crate::d3::joint`].
//!
//! In 2D the hinge is always about the implicit Z axis, so [`JointKind2D::Revolute`]
//! carries no axis (and there is no `Spherical` — that is a 3D-only 3-DOF ball
//! joint). Otherwise the families and motor/limit vocabulary mirror `d3`:
//!
//! * [`JointKind2D::Fixed`] — weld the two bodies rigidly.
//! * [`JointKind2D::Revolute`] — a hinge about Z (optional angle limits + motor).
//! * [`JointKind2D::Prismatic`] — a slider along `axis` (optional limits + motor).
//! * [`JointKind2D::Distance`] — a rope within `max_distance`.
//!
//! [`PhysicsWorld2D`]: crate::d2::PhysicsWorld2D

use glam::DVec2;
use rapier2d_f64::dynamics::{
    FixedJointBuilder, GenericJoint, JointAxis, MotorModel, PrismaticJointBuilder,
    RevoluteJointBuilder, RopeJointBuilder,
};
use rapier2d_f64::prelude::ImpulseJointHandle;

/// Opaque, stable handle to a joint in a [`PhysicsWorld2D`](crate::d2::PhysicsWorld2D).
/// See [`crate::d3::JointId3D`] for the newtype rationale.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct JointId2D(pub(crate) ImpulseJointHandle);

impl core::fmt::Debug for JointId2D {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let (i, g) = self.0.into_raw_parts();
        write!(f, "JointId2D({i}v{g})")
    }
}

impl Ord for JointId2D {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.into_raw_parts().cmp(&other.0.into_raw_parts())
    }
}
impl PartialOrd for JointId2D {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Motor parameters for a driven joint axis (revolute/prismatic). See
/// [`crate::d3::JointMotor3D`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointMotor2D {
    /// Target position (angle in radians for revolute, distance for prismatic).
    pub target_pos: f64,
    /// Target velocity.
    pub target_vel: f64,
    /// Position stiffness (`0` → a pure velocity motor).
    pub stiffness: f64,
    /// Velocity damping.
    pub damping: f64,
    /// Maximum motor force/torque.
    pub max_force: f64,
}

impl Default for JointMotor2D {
    fn default() -> Self {
        Self {
            target_pos: 0.0,
            target_vel: 0.0,
            stiffness: 0.0,
            damping: 1.0,
            max_force: f64::MAX,
        }
    }
}

/// The joint family plus its per-family parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JointKind2D {
    /// Weld the two bodies rigidly.
    Fixed,
    /// A hinge about Z. Optional `[min, max]` angle limits (radians) + motor.
    Revolute {
        limits: Option<[f64; 2]>,
        motor: Option<JointMotor2D>,
    },
    /// A slider along `axis` (unit, body-local). Optional `[min, max]` distance
    /// limits + motor.
    Prismatic {
        axis: DVec2,
        limits: Option<[f64; 2]>,
        motor: Option<JointMotor2D>,
    },
    /// A rope: the two anchors are kept within `max_distance` of each other.
    Distance { max_distance: f64 },
}

/// Description of a joint linking two bodies: the [`JointKind2D`] and the anchor
/// point on each body (its local frame).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointDesc2D {
    /// The family + its parameters.
    pub kind: JointKind2D,
    /// Anchor on the first body, in its local frame.
    pub local_anchor1: DVec2,
    /// Anchor on the second body, in its local frame.
    pub local_anchor2: DVec2,
}

impl JointDesc2D {
    /// A joint of `kind` anchored at each body's origin.
    pub fn new(kind: JointKind2D) -> Self {
        Self {
            kind,
            local_anchor1: DVec2::ZERO,
            local_anchor2: DVec2::ZERO,
        }
    }

    /// Set the anchor on the first body (its local frame).
    pub fn local_anchor1(mut self, anchor: DVec2) -> Self {
        self.local_anchor1 = anchor;
        self
    }

    /// Set the anchor on the second body (its local frame).
    pub fn local_anchor2(mut self, anchor: DVec2) -> Self {
        self.local_anchor2 = anchor;
        self
    }

    /// Lower this description to a rapier [`GenericJoint`].
    pub(crate) fn to_generic(self) -> GenericJoint {
        let a1 = self.local_anchor1;
        let a2 = self.local_anchor2;
        match self.kind {
            JointKind2D::Fixed => FixedJointBuilder::new()
                .local_anchor1(a1)
                .local_anchor2(a2)
                .into(),
            JointKind2D::Revolute { limits, motor } => {
                let mut b = RevoluteJointBuilder::new()
                    .local_anchor1(a1)
                    .local_anchor2(a2);
                if let Some([lo, hi]) = limits {
                    b = b.limits([lo, hi]);
                }
                if let Some(m) = motor {
                    b = b
                        .motor_model(MotorModel::AccelerationBased)
                        .motor(m.target_pos, m.target_vel, m.stiffness, m.damping)
                        .motor_max_force(m.max_force);
                }
                b.into()
            }
            JointKind2D::Prismatic {
                axis,
                limits,
                motor,
            } => {
                // The prismatic builder has no combined `.motor()` (unlike
                // revolute), so drive the free linear axis on the GenericJoint.
                let mut joint: GenericJoint = PrismaticJointBuilder::new(axis.normalize_or_zero())
                    .local_anchor1(a1)
                    .local_anchor2(a2)
                    .into();
                if let Some([lo, hi]) = limits {
                    joint.set_limits(JointAxis::LinX, [lo, hi]);
                }
                if let Some(m) = motor {
                    joint
                        .set_motor_model(JointAxis::LinX, MotorModel::AccelerationBased)
                        .set_motor(
                            JointAxis::LinX,
                            m.target_pos,
                            m.target_vel,
                            m.stiffness,
                            m.damping,
                        )
                        .set_motor_max_force(JointAxis::LinX, m.max_force);
                }
                joint
            }
            JointKind2D::Distance { max_distance } => RopeJointBuilder::new(max_distance)
                .local_anchor1(a1)
                .local_anchor2(a2)
                .into(),
        }
    }
}
