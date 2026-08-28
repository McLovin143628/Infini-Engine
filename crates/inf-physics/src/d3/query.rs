//! Scene-query result types. The `d3` mirror of [`crate::d2`]'s query types.

use glam::DVec3;

use super::ColliderId3D;

/// The result of a successful ray cast.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit3D {
    /// The collider the ray hit.
    pub collider: ColliderId3D,
    /// The world-space point of impact.
    pub point: DVec3,
    /// The surface normal at the impact point.
    pub normal: DVec3,
    /// The distance along the (normalized) ray direction to the impact — i.e. the
    /// time of impact, in world units.
    pub toi: f64,
}

/// The result of a successful **shape cast** — a swept-volume query (P29.3).
///
/// The `d3` sibling of [`RayHit3D`], and deliberately the same vocabulary: a
/// collider, a world point, a world normal, and a distance along the (normalized)
/// sweep direction. A shape cast is a ray cast with a body, and a caller that
/// already reads a `RayHit3D` should not have to learn a second shape of answer.
///
/// The one field a ray has no need of is [`started_penetrating`](Self::started_penetrating).
/// A ray either hits or does not; a swept *volume* can begin its sweep already
/// overlapping something, and that is not the same answer as "hit at distance 0".
/// It is the answer a clearance probe cares about most — "you cannot stand up
/// here because you are *already* inside the ceiling" — and parry reports its
/// witness point and normal as unreliable in that case (`ShapeCastStatus::
/// PenetratingOrWithinTargetDist`), so a caller that reads `normal` without
/// reading this flag is reading a number parry does not stand behind.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShapeHit3D {
    /// The collider the swept shape hit.
    pub collider: ColliderId3D,
    /// The world-space witness point on the **hit collider** at the time of
    /// impact. Unreliable when [`started_penetrating`](Self::started_penetrating).
    pub point: DVec3,
    /// The world-space outward normal of the **hit collider** at the impact.
    /// Unreliable when [`started_penetrating`](Self::started_penetrating).
    pub normal: DVec3,
    /// Distance travelled along the (normalized) sweep direction before contact,
    /// in world units. `0.0` when the sweep started in contact.
    pub toi: f64,
    /// The swept shape was **already overlapping** this collider at the start of
    /// the sweep (parry's `PenetratingOrWithinTargetDist`). `point`/`normal` are
    /// not to be trusted; `toi` is `0.0`.
    pub started_penetrating: bool,
}

/// Which **class** of collider a filtered shape cast may hit (P29.4).
///
/// Not a wire enum — it never reaches a file, so the freeze-pin law has nothing
/// to say about it — and deliberately a variant per *question a caller in this
/// repository actually asks* rather than a matrix of every combination rapier's
/// `QueryFilter` can express. The day a new caller asks a new question it gets a
/// new variant, because a knob nobody turns documents a choice nobody made.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CastTargets {
    /// Everything. What [`super::PhysicsWorld3D::cast_shape`] does, and what a
    /// clearance probe and a camera both need.
    ///
    /// # Everything includes SENSORS, and that half of P29.7's bound is still open
    ///
    /// A trigger volume is a collider with `is_sensor` set, and
    /// `QueryFilter::default()` does **not** carry `EXCLUDE_SENSORS` — so a
    /// region that exerts no force can block, or hold up, any caller of this
    /// variant. Today they are the character mover's step-up and ceiling probes
    /// and the traversal fit check (through
    /// [`super::PhysicsWorld3D::cast_shape`], which passes `All`), and
    /// `super::camera`'s occlusion sweep (through
    /// [`cast_shape_where`](super::PhysicsWorld3D::cast_shape_where)).
    ///
    /// Island wave VEH1a took the one caller whose consequence was measurable —
    /// a wheel *resting* on a checkpoint — and gave it
    /// [`AllSolid`](Self::AllSolid). It deliberately changed nothing else,
    /// because a camera that stops at a trigger and a mover that steps onto one
    /// are each a behaviour question with their own arms. **The record lives
    /// here rather than in the ROADMAP entry that used to hold it**, which was
    /// struck when the wheel half closed (VEH1a audit).
    ///
    /// [`Fixed`](Self::Fixed) is no escape either: `exclude_dynamic` says
    /// nothing about `is_sensor`, so a **static** trigger is visible to the
    /// mantle probe too.
    #[default]
    All,
    /// Static and kinematic geometry only — the broad phase leaves dynamic
    /// bodies out entirely.
    ///
    /// The mantle probe's filter. A ledge that is a crate somebody can shove is
    /// not a ledge, and the exclusion has to happen in the broad phase rather
    /// than after the cast: the P22.3 audit's M4 is a whole building that
    /// collapsed because a downstream check turned "the rubble is not support"
    /// into "the rubble HIDES support".
    Fixed,
    /// Every body kind, **sensors excluded** — the third question (island wave
    /// VEH1a), and the one a thing that must *rest* on what it hits asks.
    ///
    /// [`All`](Self::All) is "everything the broad phase holds", and a sensor is
    /// in it: a trigger volume is a collider with `is_sensor` set, so a wheel ray
    /// cast with `All` finds the top face of a checkpoint volume and the
    /// suspension pushes off it. P29.7 carried that as a named bound — *"a car
    /// crossing a trigger volume would ride on it"* — and this is its filter.
    ///
    /// It is **not** [`Fixed`](Self::Fixed): a car drives over a crate, up a
    /// fractured chunk and onto a moving platform, and all three are dynamic. The
    /// axis that matters to a suspension is *solid or not*, which is a different
    /// axis from *static or not*, and rapier spells them with two different
    /// `QueryFilterFlags`. Like `Fixed`, the exclusion is that flag and not a
    /// downstream check — `QueryFilterFlags::test` runs as the tree is walked, so
    /// a rejected sensor never becomes the nearest hit and the ground beneath it,
    /// which is what the wheel was asking about, is still found.
    AllSolid,
}
