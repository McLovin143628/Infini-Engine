//! The [`ModWorld`] embedder boundary (P14.5, deliverable 1).
//!
//! A mod never touches the engine's world directly. Every host function is a
//! method on [`ModWorld`], which the embedder implements: the runtime/editor
//! provide a real one over the `inf-ecs` world (see `inf-player`'s
//! `mods::ModWorldAdapter`), and tests use [`MockModWorld`]. This is the exact
//! analogue of the blueprint interpreter's `Host` trait — the pure sandbox on
//! one side, the engine on the other, one narrow, auditable seam between.
//!
//! Entities are opaque `i64` ids, the *same* identity space blueprints use (the
//! sim assigns each actor a stable `i64` in `Guid` order), so a mod and a
//! blueprint address the same entity by the same number.

use std::collections::{BTreeMap, BTreeSet};

/// The capability-scoped host surface a mod can reach, implemented by the
/// embedder. Each method corresponds 1:1 to a host import; the sandbox only
/// *calls* a method if the mod was granted the matching capability (an ungranted
/// import never links, so the method is never reached), but a correct embedder
/// implements them all — capability enforcement lives in the linker, not here.
pub trait ModWorld {
    /// Emit a log line (capability `log`). The default forwards to `tracing`.
    fn log(&mut self, message: &str) {
        tracing::info!(target: "inf_wasm_mod", "{message}");
    }

    /// World-space translation of `entity`, or `None` if it does not exist
    /// (capability `entities`).
    fn entity_translation(&mut self, entity: i64) -> Option<[f64; 3]>;

    /// Set the world-space translation of `entity` (capability `entities`). A
    /// non-existent entity is a silent no-op.
    fn set_entity_translation(&mut self, entity: i64, translation: [f64; 3]);

    /// Whether input action `key` is currently held (capability `input`).
    fn input_is_down(&mut self, key: &str) -> bool;

    /// Spawn a unit cube at `(x, y, z)`, returning its new opaque entity id
    /// (capability `spawn`).
    fn spawn_cube(&mut self, x: f64, y: f64, z: f64) -> i64;
}

/// An in-memory [`ModWorld`] for tests and headless proofs: entities are a
/// `BTreeMap<i64,[f64;3]>`, input is a held-key set, logs and spawns are
/// recorded. Deterministic (sorted map) so a mod's effect is reproducible.
#[derive(Debug, Default, Clone)]
pub struct MockModWorld {
    /// Entity id → translation.
    pub entities: BTreeMap<i64, [f64; 3]>,
    /// Currently-held input actions.
    pub input: BTreeSet<String>,
    /// Every line a mod logged, in order.
    pub logs: Vec<String>,
    /// Next id `spawn_cube` will hand out.
    pub next_id: i64,
    /// Ids `spawn_cube` created, in order.
    pub spawned: Vec<i64>,
}

impl MockModWorld {
    /// A world seeded with entities at the given `(id, translation)` positions.
    pub fn with_entities<I>(entities: I) -> Self
    where
        I: IntoIterator<Item = (i64, [f64; 3])>,
    {
        let entities: BTreeMap<i64, [f64; 3]> = entities.into_iter().collect();
        let next_id = entities.keys().copied().max().unwrap_or(0) + 1;
        Self {
            entities,
            next_id,
            ..Default::default()
        }
    }

    /// Mark `key` held (builder-style).
    pub fn press(mut self, key: impl Into<String>) -> Self {
        self.input.insert(key.into());
        self
    }
}

impl ModWorld for MockModWorld {
    fn log(&mut self, message: &str) {
        self.logs.push(message.to_owned());
    }

    fn entity_translation(&mut self, entity: i64) -> Option<[f64; 3]> {
        self.entities.get(&entity).copied()
    }

    fn set_entity_translation(&mut self, entity: i64, translation: [f64; 3]) {
        if let Some(slot) = self.entities.get_mut(&entity) {
            *slot = translation;
        }
    }

    fn input_is_down(&mut self, key: &str) -> bool {
        self.input.contains(key)
    }

    fn spawn_cube(&mut self, x: f64, y: f64, z: f64) -> i64 {
        let id = self.next_id.max(1);
        self.next_id = id + 1;
        self.entities.insert(id, [x, y, z]);
        self.spawned.push(id);
        id
    }
}
