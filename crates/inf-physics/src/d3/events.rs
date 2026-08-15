//! Contact/sensor events, collected per step and drained by the facade. The `d3`
//! mirror of [`crate::d2`]'s events — sharing the dimension-agnostic
//! [`ContactPhase`](crate::d2::ContactPhase).

use std::sync::Mutex;

use rapier3d_f64::prelude::{ColliderSet, CollisionEvent, ContactPair, EventHandler, RigidBodySet};

use super::ColliderId3D;
use crate::d2::ContactPhase;

/// A collision or sensor event between two colliders.
///
/// The pair is canonicalized so `collider_a <= collider_b`, and the facade sorts
/// batches of these, so the same physical situation always reports identically
/// regardless of rapier's internal ordering. `sensor` is `true` when at least one
/// collider is a sensor — i.e. this is a trigger overlap rather than a solid
/// contact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ContactEvent3D {
    /// The lower-handle collider of the pair.
    pub collider_a: ColliderId3D,
    /// The higher-handle collider of the pair.
    pub collider_b: ColliderId3D,
    /// Whether the pair started or stopped touching.
    pub phase: ContactPhase,
    /// `true` if at least one collider is a sensor (a trigger overlap).
    pub sensor: bool,
}

impl ContactEvent3D {
    fn from_rapier(event: CollisionEvent) -> Self {
        let (h1, h2, phase) = match event {
            CollisionEvent::Started(a, b, _) => (a, b, ContactPhase::Started),
            CollisionEvent::Stopped(a, b, _) => (a, b, ContactPhase::Stopped),
        };
        let (mut a, mut b) = (ColliderId3D(h1), ColliderId3D(h2));
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
    pub(crate) fn append_into(self, out: &mut Vec<ContactEvent3D>) {
        let events = self.events.into_inner().unwrap_or_else(|e| e.into_inner());
        out.extend(events.into_iter().map(ContactEvent3D::from_rapier));
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
        // **Recover from a poisoned lock rather than dropping the event**
        // (Hardening Wave C, L6.F10). `append_into`, twenty lines up, already
        // does exactly this with `into_inner()`; this half silently discarded a
        // contact instead. Poisoning means some earlier holder panicked, and the
        // buffer it panicked over is a `Vec<CollisionEvent>` with no invariant
        // to violate half-way — so the data is sound and the only thing the
        // `if let` bought was a step whose collision set is a function of
        // whether an unrelated panic had happened. That is a determinism defect
        // dressed as defensiveness: it makes `state_bytes` depend on the
        // process's history rather than on its inputs.
        let mut guard = self.events.lock().unwrap_or_else(|e| e.into_inner());
        guard.push(event);
    }

    fn handle_contact_force_event(
        &self,
        _dt: f64,
        _bodies: &RigidBodySet,
        _colliders: &ColliderSet,
        _contact_pair: &ContactPair,
        _total_force_magnitude: f64,
    ) {
        // Contact-force events are not part of the P9.1b facade surface.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rapier3_f64::prelude::CollisionEventFlags;

    /// **A contact survives a poisoned buffer** (Hardening Wave C, L6.F10).
    ///
    /// The handler used to write `if let Ok(mut guard) = self.events.lock()`,
    /// which drops the event when the mutex is poisoned — while
    /// [`EventCollector::append_into`], twenty lines up, already recovers with
    /// `into_inner()`. Poisoning means some earlier holder panicked, and the
    /// buffer it panicked over is a `Vec<CollisionEvent>` with no invariant to
    /// violate half-way, so the data is sound. What the `if let` bought was a
    /// step whose collision set is a function of **whether an unrelated panic
    /// had happened** — which makes `state_bytes` depend on the process's
    /// history rather than on its inputs. That is a determinism defect wearing
    /// defensiveness as a disguise.
    ///
    /// The panic below is deliberate and its message reaches the test log; it is
    /// raised on its own thread so the process-wide panic hook is left alone for
    /// every other test in this binary.
    #[test]
    fn a_contact_is_recorded_even_after_the_buffer_was_poisoned() {
        let collector = EventCollector::default();

        // Poison it exactly the way production would: a holder that panics.
        let poisoned = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _held = collector
                .events
                .lock()
                .expect("a fresh mutex is not poisoned");
            panic!("an unrelated panic, raised while the event buffer was held");
        }));
        assert!(poisoned.is_err(), "the panic did not unwind");
        assert!(
            collector.events.lock().is_err(),
            "the mutex is not poisoned, so this arm is testing nothing"
        );

        let h = ColliderSet::invalid_handle();
        collector.handle_collision_event(
            &RigidBodySet::new(),
            &ColliderSet::new(),
            CollisionEvent::Started(h, h, CollisionEventFlags::empty()),
            None,
        );

        let mut out = Vec::new();
        collector.append_into(&mut out);
        assert_eq!(
            out.len(),
            1,
            "the contact was dropped because an unrelated panic had happened              earlier in the process — the step's collision set is not a function              of its inputs"
        );
        assert_eq!(out[0].phase, ContactPhase::Started);
    }
}
