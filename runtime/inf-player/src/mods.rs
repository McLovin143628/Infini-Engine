//! Native WASM-mod loading for the standalone player (ROADMAP P14.5,
//! deliverable 3).
//!
//! [`PlayerMods`] loads every `.wasm` in the `--mods` directory through
//! [`inf_wasm_host::WasmMods`] (each granted the caps in its sibling `mod.toml`)
//! and implements the sim's [`ModHook`], ticking them per fixed step against a
//! [`ModWorldAdapter`] over the real [`EcsWorld`]. This module is native-only
//! (`cfg(not(target_arch = "wasm32"))`) — the browser player loads no native
//! mods, so it never pulls `wasmtime`.

use std::collections::BTreeMap;
use std::path::Path;

use uuid::Uuid;

use inf_ecs::components::{MeshRef, Primitive, Transform};
use inf_ecs::EcsWorld;
use inf_wasm_host::{ExecLimits, ModWorld, WasmMods};

use crate::runtime_sim::{ModHook, RuntimeInput};

/// High bits of the deterministic Guid a `spawn_cube` mod-spawned entity gets
/// (so replays are reproducible — no `Uuid::new_v4`).
const SPAWN_GUID_PREFIX: u128 = 0x0DDF_ACE0_0000_0000_0000_0000_0000_0000;

/// The player's loaded WASM mods, ticked each fixed step.
pub struct PlayerMods {
    session: WasmMods,
    /// The next opaque `i64` id `spawn_cube` will hand out (kept above the
    /// actor-id range so it never collides).
    next_spawn_id: i64,
}

impl PlayerMods {
    /// Load every `.wasm` in `dir`. `first_free_id` is the lowest entity id not
    /// already used by a scene actor. A mod that fails to load is skipped +
    /// reported (never fatal). Returns `Err` only if the sandbox engine itself
    /// cannot be built.
    pub fn load(dir: &Path, first_free_id: i64) -> Result<Self, String> {
        let mut session = WasmMods::new(ExecLimits::default()).map_err(|e| e.to_string())?;
        let loaded = session.load_dir(dir).map_err(|e| e.to_string())?;
        for note in session.reports() {
            tracing::warn!(target: "inf_player_mods", "{note}");
        }
        tracing::info!(
            target: "inf_player_mods",
            "loaded {loaded} mod(s) from {}",
            dir.display()
        );
        Ok(Self {
            session,
            next_spawn_id: first_free_id.max(1),
        })
    }

    /// Whether any mods were loaded.
    pub fn is_empty(&self) -> bool {
        self.session.mods().is_empty()
    }

    /// How many mods are still enabled (not disabled by a trap).
    pub fn enabled_count(&self) -> usize {
        self.session.enabled_count()
    }
}

impl ModHook for PlayerMods {
    fn tick(
        &mut self,
        world: &mut EcsWorld,
        entities: &mut BTreeMap<i64, Uuid>,
        dt: f64,
        input: &RuntimeInput,
    ) {
        let mut adapter = ModWorldAdapter {
            world,
            entities,
            input,
            next_spawn_id: &mut self.next_spawn_id,
        };
        self.session.tick(&mut adapter, dt);
    }
}

/// The [`ModWorld`] the player hands the sandbox each tick: maps opaque `i64`
/// mod entity ids through the blueprint `entities` map onto the real ECS world.
struct ModWorldAdapter<'a> {
    world: &'a mut EcsWorld,
    entities: &'a mut BTreeMap<i64, Uuid>,
    input: &'a RuntimeInput,
    next_spawn_id: &'a mut i64,
}

impl ModWorld for ModWorldAdapter<'_> {
    fn log(&mut self, message: &str) {
        tracing::info!(target: "inf_player_mods", "{message}");
    }

    fn entity_translation(&mut self, entity: i64) -> Option<[f64; 3]> {
        let guid = *self.entities.get(&entity)?;
        let e = self.world.entity_of(guid)?;
        let t = self.world.world().get::<Transform>(e)?;
        Some([t.translation.x, t.translation.y, t.translation.z])
    }

    fn set_entity_translation(&mut self, entity: i64, translation: [f64; 3]) {
        let Some(&guid) = self.entities.get(&entity) else {
            return;
        };
        let Some(e) = self.world.entity_of(guid) else {
            return;
        };
        if let Some(mut t) = self.world.world_mut().get_mut::<Transform>(e) {
            t.translation.x = translation[0];
            t.translation.y = translation[1];
            t.translation.z = translation[2];
            self.world.mark_dirty();
        }
    }

    fn input_is_down(&mut self, key: &str) -> bool {
        self.input.is_down(key)
    }

    fn spawn_cube(&mut self, x: f64, y: f64, z: f64) -> i64 {
        let id = *self.next_spawn_id;
        *self.next_spawn_id += 1;
        // Deterministic Guid derived from the id (replay-safe).
        let guid = Uuid::from_u128(SPAWN_GUID_PREFIX | (id as u128));
        let e = self
            .world
            .spawn_with_guid(guid, &format!("Mod Cube {id}"), None);
        {
            let w = self.world.world_mut();
            if let Some(mut t) = w.get_mut::<Transform>(e) {
                t.translation.x = x;
                t.translation.y = y;
                t.translation.z = z;
            }
            w.entity_mut(e).insert(MeshRef {
                primitive: Primitive::Cube,
                ..Default::default()
            });
        }
        self.entities.insert(id, guid);
        self.world.mark_dirty();
        id
    }
}
