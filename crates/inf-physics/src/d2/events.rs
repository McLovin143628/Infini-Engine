//! Contact/sensor events, collected per step and drained by the facade.

use std::sync::Mutex;

use rapier2d_f64::prelude::{ColliderSet, CollisionEvent, ContactPair, EventHandler, RigidBodySet};

use super::ColliderId;

/// Whether a contact/overlap began or ended this step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ContactPhase {
    /// The pair started touching (or a sensor started overlapping) this step.
    Started,
    /// The pair stopped touching (or a sensor stopped overlapping) this step.
    Stopped,
}

/// A collision or sensor event between two colliders.
///
/// The pair is canonicalized so `collider_a <= collider_b`, and the facade sorts
/// batches of these, so the same physical situation always reports identically
/// regardless of rapier's internal ordering. `sensor` is `true` when at least one
/// collider is a sensor — i.e. this is a trigger overlap rather than a solid
/// contact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ContactEvent2D {
    /// The lower-handle collider of the pair.
    pub collider_a: ColliderId,
    /// The higher-handle collider of the pair.
    pub collider_b: ColliderId,
    /// Whether the pair started or stopped touching.
    pub phase: ContactPhase,
    /// `true` if at least one collider is a sensor (a trigger overlap).
    pub sensor: bool,
}

impl ContactEvent2D {
    fn from_rapier(event: CollisionEvent) -> Self {
        let (h1, h2, phase) = match event {
            CollisionEvent::Started(a, b, _) => (a, b, ContactPhase::Started),
            CollisionEvent::Stopped(a, b, _) => (a, b, ContactPhase::Stopped),
        };
        let (mut a, mut b) = (ColliderId(h1), ColliderId(h2));
        if b < a {
            std::mem::swap(&mut a, &mut b);
        }
        Self {
            collider_a: a,
            collider_b: b,
            phase,
            sensor: event.sensor(),
        }
    }
}

/// A rapier [`EventHandler`] that buffers collision events during a step.
///
/// It uses a `Mutex` because the trait requires `Send + Sync`; with rapier's
/// `parallel` feature off (our determinism choice) there is never real contention,
/// so the lock is uncontended and does not affect reproducibility.
#[derive(Default)]
pub(crate) struct EventCollector {
    events: Mutex<Vec<CollisionEvent>>,
}

impl EventCollector {
    /// Convert the buffered rapier events into facade events and append them to
    /// `out`. Ordering is imposed later by the drain (`sort`), not here.
    pub(crate) fn append_into(self, out: &mut Vec<ContactEvent2D>) {
        let events = self.events.into_inner().unwrap_or_else(|e| e.into_inner());
        out.extend(events.into_iter().map(ContactEvent2D::from_rapier));
    }
}

impl EventHandler for EventCollector {
    fn handle_collision_event(
        &self,
        _bodies: &RigidBodySet,
        _colliders: &ColliderSet,
        event: CollisionEvent,
        _contact_pair: Option<&ContactPair>,
    ) {
        if let Ok(mut guard) = self.events.lock() {
            guard.push(event);
        }
    }

    fn handle_contact_force_event(
        &self,
        _dt: f64,
        _bodies: &RigidBodySet,
        _colliders: &ColliderSet,
        _contact_pair: &ContactPair,
        _total_force_magnitude: f64,
    ) {
        // Contact-force events are not part of the P8.3a facade surface.
    }
}
