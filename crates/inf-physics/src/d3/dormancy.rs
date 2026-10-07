//! **STATIC SCENERY SLEEPS WHERE NOTHING CAN TOUCH IT** (wave PERF1b).
//!
//! # The measurement that put this here
//!
//! Profiled in the shipped window on the cooked island (`INF_STEP_LOG`), the
//! rapier step cost 2.7 ms with the hero standing, 4.4 walking and 5.0-5.3
//! running -- and **96 % of the 58 000-103 000 pairs rapier tracked were
//! fixed-fixed**: a wall touching the wall beside it, a floor slab inside a
//! terrain tile's box, a kerb along a street. rapier 0.34's BVH broad phase
//! makes a pair for every overlapping pair of AABBs whatever the bodies are,
//! and every step it walks the whole pair map (`BroadPhaseBvh::update`'s
//! retain) and every contact-graph edge (`NarrowPhase::compute_contacts`, whose
//! own source says "TODO PERF: don't iterate on all the edges"). A static
//! solid in this engine carries `ActiveCollisionTypes::all() - FIXED_FIXED`,
//! so not one of those pairs can ever produce a contact or an event: they were
//! pure bookkeeping, paid every step, in proportion to how much city was in
//! the band.
//!
//! # The rule
//!
//! A static collider takes part in rapier's broad phase only while some
//! **mover** -- a collider on a dynamic or kinematic body -- is within reach of
//! it: its AABB inflated by [`MARGIN_M`] plus what the body can travel in two
//! steps. Out of reach it is disabled in rapier (and so leaves the broad phase,
//! taking its fixed-fixed pairs with it) and is **still in the query tree**:
//! every ray, shape cast, AABB test and the character mover sees it exactly as
//! before, because [`PhysicsWorld3D`](super::PhysicsWorld3D)'s filtered query
//! doors treat a sleeping static as enabled and the unfiltered ones never
//! tested the flag.
//!
//! What a sleeping static loses is exactly its pairs with other statics, which
//! were inactive. A pair with a mover is created when the two AABBs meet, as
//! before, because the static was woken while the mover was still a margin
//! away. Exempt, and never put to sleep: sensors (a trigger is a different
//! contract), any collider whose own pairing includes `FIXED_FIXED`, a collider
//! a caller has switched on or off by hand (`set_collider_enabled`'s parked
//! capsules), and a static whose body has been moved after it was attached
//! (a moving static is unusual and is left exactly as it always was).
//!
//! # Determinism
//!
//! A pure function of the physics world's own state at the start of the step:
//! the movers' poses and velocities and the statics' boxes. Every walk is in
//! handle order (`BTreeMap` / `BTreeSet` over raw handle parts) and the
//! enable / disable calls are applied sorted, so both hosts -- which step the
//! same world through the same door -- make the same calls in the same order.
//!
//! # Cost
//!
//! Incremental: each mover keeps a **fat box** ([`FAT_M`] past what it needs)
//! and re-queries the tree only when what it needs leaves it -- a walking
//! agent every few seconds, a car every few dozen steps. A static that arrives
//! (a band re-description) is tested once against the movers' fat boxes
//! through a coarse grid.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use rapier3d_f64::parry::bounding_volume::{Aabb, BoundingVolume};
use rapier3d_f64::prelude::{ActiveCollisionTypes, ColliderHandle, RigidBodyHandle};

/// The reach a mover keeps statics awake within, beyond what it can travel in
/// two steps, in metres. Generous against rapier's own prediction distance
/// (millimetres) and the broad phase's change-detection skin.
pub const MARGIN_M: f64 = 0.75;

/// How far past what it needs a mover's fat box reaches, in metres -- the
/// slack that lets a mover move for a while without re-querying.
pub const FAT_M: f64 = 4.0;

/// The coarse grid a batch of arriving statics is tested through, in metres.
const GRID_M: f64 = 32.0;

type Key = (u32, u32);

fn ckey(h: ColliderHandle) -> Key {
    h.into_raw_parts()
}

fn bkey(h: RigidBodyHandle) -> Key {
    h.into_raw_parts()
}

#[derive(Debug, Clone)]
struct StaticRec {
    handle: ColliderHandle,
    /// How many movers' fat boxes overlap this static's box.
    cover: u32,
    /// Disabled in rapier by this module (and only by it).
    asleep: bool,
}

#[derive(Debug, Clone)]
struct MoverRec {
    fat: Aabb,
    /// The managed statics this mover counted, by key (may hold keys of
    /// statics since removed; a decrement skips those).
    covered: Vec<Key>,
}

/// What the dormancy has done, for arms and diagnostics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DormancyStats {
    /// Statics under management (neither exempt nor a mover's).
    pub managed: usize,
    /// Of those, how many are asleep right now.
    pub asleep: usize,
    /// Movers tracked.
    pub movers: usize,
    /// Enable / disable calls made since the world was built.
    pub toggles: u64,
    /// Fat-box re-queries since the world was built.
    pub requeries: u64,
}

/// The per-world state; see the module docs.
#[derive(Debug, Default)]
pub(crate) struct Dormancy {
    /// `true` turns the whole mechanism off (every static awake) -- the A/B
    /// door the arms compare against.
    pub(crate) off: bool,
    statics: BTreeMap<Key, StaticRec>,
    movers: BTreeMap<Key, MoverRec>,
    /// Bodies that are dynamic or kinematic.
    mover_bodies: BTreeMap<Key, RigidBodyHandle>,
    /// Statics attached since the last update.
    fresh: Vec<ColliderHandle>,
    /// Colliders never managed: hand-toggled, or on a static that moved.
    exempt: BTreeSet<Key>,
    /// Keys whose cover crossed zero (or arrived) this update.
    touched: Vec<Key>,
    toggles: u64,
    requeries: u64,
}

impl Dormancy {
    pub(crate) fn body_added(&mut self, h: RigidBodyHandle, mover: bool) {
        if mover {
            self.mover_bodies.insert(bkey(h), h);
        }
    }

    /// A body changed kind. Its colliders change side: a body that became a
    /// mover stops being scenery (woken, unmanaged); one that became fixed has
    /// its colliders offered as fresh statics.
    pub(crate) fn body_kind_changed(
        &mut self,
        h: RigidBodyHandle,
        mover: bool,
        colliders: &[ColliderHandle],
        set: &mut rapier3d_f64::prelude::ColliderSet,
    ) {
        if mover {
            self.mover_bodies.insert(bkey(h), h);
            for c in colliders {
                self.unmanage(*c, set);
            }
        } else {
            self.mover_bodies.remove(&bkey(h));
            self.drop_mover(bkey(h));
            self.fresh.extend_from_slice(colliders);
        }
    }

    pub(crate) fn body_removed(&mut self, h: RigidBodyHandle, colliders: &[ColliderHandle]) {
        self.mover_bodies.remove(&bkey(h));
        self.drop_mover(bkey(h));
        for c in colliders {
            self.statics.remove(&ckey(*c));
            self.exempt.remove(&ckey(*c));
        }
    }

    pub(crate) fn collider_added(&mut self, c: ColliderHandle, parent_is_fixed: bool) {
        if parent_is_fixed {
            self.fresh.push(c);
        }
    }

    pub(crate) fn collider_removed(&mut self, c: ColliderHandle) {
        self.statics.remove(&ckey(c));
        self.exempt.remove(&ckey(c));
    }

    /// A caller toggled this collider by hand: it is theirs from now on.
    pub(crate) fn hand_toggled(&mut self, c: ColliderHandle) {
        self.exempt.insert(ckey(c));
        self.statics.remove(&ckey(c));
    }

    /// A fixed body moved: its colliders leave management, awake.
    pub(crate) fn static_moved(
        &mut self,
        colliders: &[ColliderHandle],
        set: &mut rapier3d_f64::prelude::ColliderSet,
    ) {
        for c in colliders {
            if self.statics.contains_key(&ckey(*c)) {
                self.unmanage(*c, set);
                self.exempt.insert(ckey(*c));
            }
        }
    }

    /// Whether this collider is disabled in rapier BY THIS MODULE -- the
    /// filtered query doors treat such a collider as enabled.
    pub(crate) fn is_asleep(&self, c: ColliderHandle) -> bool {
        self.statics.get(&ckey(c)).is_some_and(|s| s.asleep)
    }

    pub(crate) fn stats(&self) -> DormancyStats {
        DormancyStats {
            managed: self.statics.len(),
            asleep: self.statics.values().filter(|s| s.asleep).count(),
            movers: self.movers.len(),
            toggles: self.toggles,
            requeries: self.requeries,
        }
    }

    fn unmanage(&mut self, c: ColliderHandle, set: &mut rapier3d_f64::prelude::ColliderSet) {
        if let Some(rec) = self.statics.remove(&ckey(c)) {
            if rec.asleep {
                if let Some(col) = set.get_mut(c) {
                    col.set_enabled(true);
                    self.toggles += 1;
                }
            }
        }
    }

    fn drop_mover(&mut self, k: Key) {
        if let Some(m) = self.movers.remove(&k) {
            for s in m.covered {
                self.decrement(s);
            }
        }
    }

    fn decrement(&mut self, s: Key) {
        if let Some(rec) = self.statics.get_mut(&s) {
            rec.cover = rec.cover.saturating_sub(1);
            if rec.cover == 0 {
                self.touched.push(s);
            }
        }
    }

    fn increment(&mut self, s: Key) {
        if let Some(rec) = self.statics.get_mut(&s) {
            rec.cover += 1;
            if rec.cover == 1 {
                self.touched.push(s);
            }
        }
    }

    /// Wake everything this module put to sleep and forget it all (the A/B
    /// door being switched off mid-session).
    pub(crate) fn wake_all(&mut self, set: &mut rapier3d_f64::prelude::ColliderSet) {
        let keys: Vec<Key> = self.statics.keys().copied().collect();
        for k in keys {
            let h = self.statics[&k].handle;
            self.unmanage(h, set);
        }
        self.movers.clear();
        self.fresh.clear();
    }
}

/// Is this collider scenery this module may put to sleep?
fn manageable(col: &rapier3d_f64::prelude::Collider) -> bool {
    !col.is_sensor()
        && !col
            .active_collision_types()
            .contains(ActiveCollisionTypes::FIXED_FIXED)
}

fn inflate(a: &Aabb, by: f64) -> Aabb {
    a.loosened(by)
}

fn cells(a: &Aabb) -> impl Iterator<Item = (i64, i64)> {
    let x0 = (a.mins.x / GRID_M).floor() as i64;
    let x1 = (a.maxs.x / GRID_M).floor() as i64;
    let z0 = (a.mins.z / GRID_M).floor() as i64;
    let z1 = (a.maxs.z / GRID_M).floor() as i64;
    (x0..=x1).flat_map(move |x| (z0..=z1).map(move |z| (x, z)))
}

type Query<'q> = dyn FnMut(&Aabb, &rapier3d_f64::prelude::ColliderSet) -> Vec<ColliderHandle> + 'q;

/// One update, at the start of a step, before rapier's pipeline runs. See the
/// module docs. `query` answers the statics overlapping a box from the world's
/// query tree.
pub(crate) fn update(
    d: &mut Dormancy,
    bodies: &rapier3d_f64::prelude::RigidBodySet,
    colliders: &mut rapier3d_f64::prelude::ColliderSet,
    dt: f64,
    query: &mut Query<'_>,
) {
    if d.off {
        return;
    }
    // 1. Movers: a body that is no longer one (or is gone, or has nothing
    //    enabled to touch with) drops what it covered; one whose need left its
    //    fat box re-queries.
    requery_movers(d, bodies, colliders, dt, query);
    // 2. Fresh statics: managed from now on, covered by whichever movers' fat
    //    boxes reach them AFTER this update's re-queries -- so a static arriving
    //    inside a box that just moved is counted against the box it is in, and
    //    no re-query has to find a leaf the tree may not hold yet. A static that
    //    cannot be managed (a sensor, or a pairing that includes `FIXED_FIXED`)
    //    keeps the statics around it awake instead: its own pairs with them are
    //    ACTIVE -- a trigger volume over a wall reports -- so its body becomes a
    //    mover.
    let pinned = admit_fresh(d, bodies, colliders);
    if pinned {
        requery_movers(d, bodies, colliders, dt, query);
    }
    // 3. Apply: wake what is covered, sleep what is not, sorted by handle.
    let mut touched = std::mem::take(&mut d.touched);
    touched.sort_unstable();
    touched.dedup();
    for k in touched {
        let Some(rec) = d.statics.get_mut(&k) else {
            continue;
        };
        let want_asleep = rec.cover == 0;
        if want_asleep == rec.asleep {
            continue;
        }
        let Some(col) = colliders.get_mut(rec.handle) else {
            d.statics.remove(&k);
            continue;
        };
        col.set_enabled(!want_asleep);
        rec.asleep = want_asleep;
        d.toggles += 1;
    }
}

fn requery_movers(
    d: &mut Dormancy,
    bodies: &rapier3d_f64::prelude::RigidBodySet,
    colliders: &rapier3d_f64::prelude::ColliderSet,
    dt: f64,
    query: &mut Query<'_>,
) {
    let gone: Vec<Key> = d
        .movers
        .keys()
        .filter(|k| !d.mover_bodies.contains_key(k))
        .copied()
        .collect();
    for k in gone {
        d.drop_mover(k);
    }
    let movers: Vec<(Key, RigidBodyHandle)> =
        d.mover_bodies.iter().map(|(k, h)| (*k, *h)).collect();
    for (k, h) in movers {
        let Some(rb) = bodies.get(h) else {
            d.drop_mover(k);
            continue;
        };
        // **Where the body IS, not where its colliders were** (the PERF1b
        // audit). rapier re-poses a body's colliders only inside the pipeline,
        // so after `set_body_translation` -- a teleport, a dispatcher placement,
        // every kinematic pose the bridge pushes -- `Collider::compute_aabb`
        // answers at the pose of the LAST step. Measured: a ball teleported
        // against a dormant wall fell 1.6 mm through the dormant floor on its
        // first step there (`perf1b_dormancy_cases`). The box is taken at the
        // body's current pose AND its next one (a kinematic target), merged.
        let mut need: Option<Aabb> = None;
        for c in rb.colliders() {
            if let Some(col) = colliders.get(*c) {
                if col.is_enabled() || d.is_asleep(*c) {
                    let local = col.position_wrt_parent().copied().unwrap_or_default();
                    let now = col.shape().compute_aabb(&(*rb.position() * local));
                    let next = col.shape().compute_aabb(&(*rb.next_position() * local));
                    let a = now.merged(&next);
                    need = Some(need.map_or(a, |n| n.merged(&a)));
                }
            }
        }
        let Some(need) = need else {
            d.drop_mover(k);
            continue;
        };
        let radius = need.half_extents().length();
        let travel = (rb.linvel().length() + rb.angvel().length() * radius) * dt
            + (rb.next_position().translation - rb.position().translation).length();
        let need = inflate(&need, MARGIN_M + 2.0 * travel);
        if d.movers.get(&k).is_some_and(|m| m.fat.contains(&need)) {
            continue;
        }
        let fat = inflate(&need, FAT_M + 4.0 * travel);
        d.requeries += 1;
        let mut covered: Vec<Key> = query(&fat, colliders)
            .into_iter()
            .map(ckey)
            .filter(|s| d.statics.contains_key(s))
            .collect();
        covered.sort_unstable();
        covered.dedup();
        if let Some(old) = d.movers.remove(&k) {
            for s in old.covered {
                d.decrement(s);
            }
        }
        for s in &covered {
            d.increment(*s);
        }
        d.movers.insert(k, MoverRec { fat, covered });
    }
}

/// Take this update's fresh statics under management. Returns whether one of
/// them pinned its body as a mover (a sensor, or a `FIXED_FIXED` pairing).
fn admit_fresh(
    d: &mut Dormancy,
    bodies: &rapier3d_f64::prelude::RigidBodySet,
    colliders: &rapier3d_f64::prelude::ColliderSet,
) -> bool {
    let mut fresh = std::mem::take(&mut d.fresh);
    if fresh.is_empty() {
        return false;
    }
    fresh.sort_by_key(|h| ckey(*h));
    fresh.dedup();
    let mut grid: HashMap<(i64, i64), Vec<Key>> = HashMap::new();
    for (k, m) in &d.movers {
        for c in cells(&m.fat) {
            grid.entry(c).or_default().push(*k);
        }
    }
    let mut pinned = false;
    for h in fresh {
        let k = ckey(h);
        if d.exempt.contains(&k) || d.statics.contains_key(&k) {
            continue;
        }
        let Some(col) = colliders.get(h) else {
            continue;
        };
        let parent = col.parent();
        let parent_fixed = parent
            .and_then(|b| bodies.get(b))
            .is_none_or(|b| b.is_fixed());
        if !parent_fixed {
            continue;
        }
        if !manageable(col) {
            if let Some(b) = parent {
                d.mover_bodies.insert(bkey(b), b);
                pinned = true;
            }
            continue;
        }
        let aabb = col.compute_aabb();
        let mut over: Vec<Key> = Vec::new();
        for c in cells(&aabb) {
            if let Some(ms) = grid.get(&c) {
                over.extend(ms.iter().copied());
            }
        }
        over.sort_unstable();
        over.dedup();
        let mut cover = 0u32;
        for m in over {
            if let Some(rec) = d.movers.get_mut(&m) {
                if rec.fat.intersects(&aabb) {
                    rec.covered.push(k);
                    cover += 1;
                }
            }
        }
        d.statics.insert(
            k,
            StaticRec {
                handle: h,
                cover,
                asleep: false,
            },
        );
        d.touched.push(k);
    }
    pinned
}
