//! Joints (P12.1): constraints linking two bodies in a [`PhysicsWorld3D`]. The
//! `d3` sibling of [`crate::d2::joint`].
//!
//! A joint ties two rigid bodies together, removing degrees of freedom between
//! them. The facade exposes the five game-relevant families behind one flat,
//! Details-friendly [`JointDesc3D`] (a `kind` enum + shared anchor/axis/limit/
//! motor fields), each mapped to a rapier joint builder:
//!
//! * [`JointKind3D::Fixed`] — weld the two bodies rigidly.
//! * [`JointKind3D::Revolute`] — a hinge about `axis` (with optional angle limits
//!   and a motor).
//! * [`JointKind3D::Prismatic`] — a slider along `axis` (optional distance limits
//!   and a motor).
//! * [`JointKind3D::Spherical`] — a ball-and-socket (3-DOF rotation).
//! * [`JointKind3D::Distance`] — a rope: the two anchors stay within `max_distance`.
//!
//! Every joint the facade creates is an **impulse joint** (rapier's
//! `ImpulseJointSet`); we do not use reduced-coordinate multibody joints so the
//! solver state stays uniform and the determinism discipline (crate docs) holds.
//!
//! [`PhysicsWorld3D`]: crate::d3::PhysicsWorld3D

use glam::DVec3;
use rapier3d_f64::dynamics::{
    FixedJointBuilder, GenericJoint, JointAxis, MotorModel, PrismaticJointBuilder,
    RevoluteJointBuilder, RopeJointBuilder, SphericalJointBuilder,
};
use rapier3d_f64::prelude::ImpulseJointHandle;

/// Opaque, stable handle to a joint in a [`PhysicsWorld3D`](crate::d3::PhysicsWorld3D).
///
/// Wraps rapier's generational `ImpulseJointHandle` exactly as [`BodyId3D`] wraps a
/// body handle: the concrete rapier type never leaks, a destroyed handle reads
/// back as "not found" rather than aliasing a new joint, and the facade sorts
/// every returned collection by handle so iteration order is deterministic.
///
/// [`BodyId3D`]: crate::d3::BodyId3D
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct JointId3D(pub(crate) ImpulseJointHandle);

impl core::fmt::Debug for JointId3D {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let (i, g) = self.0.into_raw_parts();
        write!(f, "JointId3D({i}v{g})")
    }
}

impl Ord for JointId3D {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.into_raw_parts().cmp(&other.0.into_raw_parts())
    }
}
impl PartialOrd for JointId3D {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Motor parameters for a driven joint axis (revolute/prismatic).
///
/// A motor drives the joint toward a target: set `target_vel` (and leave
/// `stiffness == 0`) for a velocity motor, or `target_pos` with a non-zero
/// `stiffness`/`damping` for a position (spring) motor. `max_force` caps the
/// impulse the motor may apply (`f64::MAX` for "unbounded").
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointMotor3D {
    /// Target position (angle in radians for revolute, distance for prismatic).
    pub target_pos: f64,
    /// Target velocity (rad/s for revolute, units/s for prismatic).
    pub target_vel: f64,
    /// Position stiffness (spring constant). `0` → a pure velocity motor.
    pub stiffness: f64,
    /// Velocity damping.
    pub damping: f64,
    /// Maximum motor force/torque (impulse cap).
    pub max_force: f64,
}

impl Default for JointMotor3D {
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

/// The joint family plus its per-family parameters (axis, limits, motor).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JointKind3D {
    /// Weld the two bodies rigidly (all 6 DOF removed).
    Fixed,
    /// A hinge about `axis` (unit, body-local). Optional `[min, max]` angle limits
    /// (radians) and an optional motor about the hinge axis.
    Revolute {
        axis: DVec3,
        limits: Option<[f64; 2]>,
        motor: Option<JointMotor3D>,
    },
    /// A slider along `axis` (unit, body-local). Optional `[min, max]` distance
    /// limits and an optional motor along the axis.
    Prismatic {
        axis: DVec3,
        limits: Option<[f64; 2]>,
        motor: Option<JointMotor3D>,
    },
    /// A ball-and-socket allowing all rotation about the shared anchor.
    Spherical,
    /// A rope: the two anchors are kept within `max_distance` of each other.
    Distance { max_distance: f64 },
}

/// Description of a joint linking two bodies: the [`JointKind3D`] and the anchor
/// point on each body (in that body's local frame).
///
/// Build with [`JointDesc3D::new`] and the fluent setters. The two bodies
/// themselves are passed to [`add_joint`](crate::d3::PhysicsWorld3D::add_joint).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointDesc3D {
    /// The family + its parameters.
    pub kind: JointKind3D,
    /// Anchor on the first body, in its local frame.
    pub local_anchor1: DVec3,
    /// Anchor on the second body, in its local frame.
    pub local_anchor2: DVec3,
}

impl JointDesc3D {
    /// A joint of `kind` anchored at each body's origin.
    pub fn new(kind: JointKind3D) -> Self {
        Self {
            kind,
            local_anchor1: DVec3::ZERO,
            local_anchor2: DVec3::ZERO,
        }
    }

    /// Set the anchor on the first body (its local frame).
    pub fn local_anchor1(mut self, anchor: DVec3) -> Self {
        self.local_anchor1 = anchor;
        self
    }

    /// Set the anchor on the second body (its local frame).
    pub fn local_anchor2(mut self, anchor: DVec3) -> Self {
        self.local_anchor2 = anchor;
        self
    }

    /// Lower this description to a rapier [`GenericJoint`].
    pub(crate) fn to_generic(self) -> GenericJoint {
        let a1 = self.local_anchor1;
        let a2 = self.local_anchor2;
        match self.kind {
            JointKind3D::Fixed => FixedJointBuilder::new()
                .local_anchor1(a1)
                .local_anchor2(a2)
                .into(),
            JointKind3D::Revolute {
                axis,
                limits,
                motor,
            } => {
                let mut b = RevoluteJointBuilder::new(axis.normalize_or_zero())
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
            JointKind3D::Prismatic {
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
            JointKind3D::Spherical => SphericalJointBuilder::new()
                .local_anchor1(a1)
                .local_anchor2(a2)
                .into(),
            JointKind3D::Distance { max_distance } => RopeJointBuilder::new(max_distance)
                .local_anchor1(a1)
                .local_anchor2(a2)
                .into(),
        }
    }
}
