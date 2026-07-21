//! In-editor WASM mod session with hot-reload (ROADMAP P14.5, deliverable 3).
//!
//! [`ModsSession`] mirrors the player's mod loader (`inf_player::mods`) inside
//! the editor: it loads every `.wasm` in a mods directory through the same
//! [`inf_wasm_host::WasmMods`] sandbox (each granted the caps in its sibling
//! `mod.toml`) and ticks them in Simulate. A `notify` file-watcher over the mods
//! dir flags changes; [`ModsSession::poll_reload`] — called between fixed steps —
//! re-instantiates the changed modules, so editing + rebuilding a mod updates the
//! running game without leaving Simulate.
//!
//! # Integration status (honest)
//!
//! This is a **standalone, test-proven** session: [`tick`](ModsSession::tick)
//! takes a [`ModWorld`], so it drives a `MockModWorld` in tests and would drive an
//! `EcsWorld` adapter (the exact analogue of `inf_player::mods::ModWorldAdapter`)
//! in Simulate. Wiring `poll_reload` + `tick` into `SimSession::fixed_step` behind
//! an `EcsWorld` adapter is a small, documented follow-up — the reload mechanism +
//! sandbox path are proven here.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use notify::RecursiveMode;
use notify_debouncer_full::{new_debouncer, DebounceEventResult};

use inf_wasm_host::{ExecLimits, ModWorld, WasmMods};

/// A live, hot-reloadable set of editor mods over a directory.
pub struct ModsSession {
    dir: PathBuf,
    session: WasmMods,
    /// Set by the watcher when any file under the mods dir changes.
    dirty: Arc<AtomicBool>,
    /// Kept alive for its background thread; dropping it stops watching.
    _debouncer: Option<
        notify_debouncer_full::Debouncer<
            notify::RecommendedWatcher,
            notify_debouncer_full::RecommendedCache,
        >,
    >,
}

impl ModsSession {
    /// Load every `.wasm` in `dir` and start watching it for hot-reload. A mod
    /// that fails to load is skipped + reported (see
    /// [`reports`](ModsSession::reports)); the watcher failing to start is
    /// non-fatal (hot-reload is simply disabled).
    pub fn open(dir: &Path) -> Result<Self, String> {
        let mut session = WasmMods::new(ExecLimits::default()).map_err(|e| e.to_string())?;
        session.load_dir(dir).map_err(|e| e.to_string())?;

        let dirty = Arc::new(AtomicBool::new(false));
        let debouncer = Self::spawn_watcher(dir, dirty.clone());

        Ok(Self {
            dir: dir.to_path_buf(),
            session,
            dirty,
            _debouncer: debouncer,
        })
    }

    /// Start a debounced watcher that flips `dirty` on any change under `dir`.
    /// Returns `None` (hot-reload disabled) if the OS watch can't start.
    fn spawn_watcher(
        dir: &Path,
        dirty: Arc<AtomicBool>,
    ) -> Option<
        notify_debouncer_full::Debouncer<
            notify::RecommendedWatcher,
            notify_debouncer_full::RecommendedCache,
        >,
    > {
        let mut debouncer = new_debouncer(
            Duration::from_millis(250),
            None,
            move |res: DebounceEventResult| {
                if let Ok(events) = res {
                    if !events.is_empty() {
                        dirty.store(true, Ordering::Relaxed);
                    }
                }
            },
        )
        .map_err(|e| tracing::warn!("mods watcher: {e}"))
        .ok()?;
        debouncer
            .watch(dir, RecursiveMode::NonRecursive)
            .map_err(|e| tracing::warn!("mods watch {dir:?}: {e}"))
            .ok()?;
        Some(debouncer)
    }

    /// If the watcher observed a change since the last poll, reload all mods and
    /// return `true`. Called between fixed steps in Simulate.
    pub fn poll_reload(&mut self) -> bool {
        if self.dirty.swap(false, Ordering::Relaxed) {
            self.reload();
            true
        } else {
            false
        }
    }

    /// Force a reload of every mod from the dir (discarding current mod state).
    pub fn reload(&mut self) {
        if let Err(e) = self.session.reload_dir(&self.dir) {
            tracing::warn!("mods reload: {e}");
        }
    }

    /// Tick every enabled mod once against `world`.
    pub fn tick(&mut self, world: &mut dyn ModWorld, dt: f64) {
        self.session.tick(world, dt);
    }

    /// How many mods are still enabled (not disabled by a trap).
    pub fn enabled_count(&self) -> usize {
        self.session.enabled_count()
    }

    /// Diagnostic notes (load skips + runtime disables).
    pub fn reports(&self) -> &[String] {
        self.session.reports()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_wasm_host::{wat_to_wasm, MockModWorld};

    /// A mod that moves entity 1 to a fixed point — parameterized by the point so
    /// two versions differ observably (proving reload).
    fn mover_wat(x: f64, y: f64, z: f64) -> String {
        format!(
            r#"
            (module
              (import "env" "set_entity_translation"
                (func $set (param i64 f64 f64 f64)))
              (memory (export "memory") 1)
              (func (export "mod_update") (param $dt f64)
                (call $set (i64.const 1) (f64.const {x}) (f64.const {y}) (f64.const {z}))))
        "#
        )
    }

    fn write_mod(dir: &Path, wat: &str, caps_toml: &str) {
        std::fs::write(dir.join("m.wasm"), wat_to_wasm(wat).unwrap()).unwrap();
        std::fs::write(dir.join("mod.toml"), caps_toml).unwrap();
    }

    const ENTITIES_CAP: &str = "[caps]\nentities = true\n";

    #[test]
    fn loads_and_ticks_a_mod() {
        let dir = tempfile::tempdir().unwrap();
        write_mod(dir.path(), &mover_wat(5.0, 6.0, 7.0), ENTITIES_CAP);

        let mut session = ModsSession::open(dir.path()).unwrap();
        assert_eq!(session.enabled_count(), 1);

        let mut world = MockModWorld::with_entities([(1, [0.0; 3])]);
        session.tick(&mut world, 0.016);
        assert_eq!(world.entities[&1], [5.0, 6.0, 7.0]);
    }

    #[test]
    fn reload_picks_up_new_behavior() {
        let dir = tempfile::tempdir().unwrap();
        write_mod(dir.path(), &mover_wat(1.0, 1.0, 1.0), ENTITIES_CAP);

        let mut session = ModsSession::open(dir.path()).unwrap();
        let mut world = MockModWorld::with_entities([(1, [0.0; 3])]);
        session.tick(&mut world, 0.016);
        assert_eq!(world.entities[&1], [1.0, 1.0, 1.0]);

        // Rewrite the mod to move somewhere else, then hot-reload.
        write_mod(dir.path(), &mover_wat(9.0, 8.0, 7.0), ENTITIES_CAP);
        session.reload();
        session.tick(&mut world, 0.016);
        assert_eq!(world.entities[&1], [9.0, 8.0, 7.0]);
    }

    #[test]
    fn poll_reload_is_false_without_changes() {
        let dir = tempfile::tempdir().unwrap();
        write_mod(dir.path(), &mover_wat(1.0, 1.0, 1.0), ENTITIES_CAP);
        let mut session = ModsSession::open(dir.path()).unwrap();
        // No file touched since open → nothing to reload.
        assert!(!session.poll_reload());
    }

    #[test]
    fn ungranted_capability_mod_is_skipped_not_fatal() {
        let dir = tempfile::tempdir().unwrap();
        // Grants nothing, but the mod imports set_entity_translation → skipped.
        write_mod(dir.path(), &mover_wat(1.0, 1.0, 1.0), "[caps]\n");
        let session = ModsSession::open(dir.path()).unwrap();
        assert_eq!(session.enabled_count(), 0, "no mod loaded");
        assert!(
            session.reports().iter().any(|r| r.contains("entities")),
            "reports the missing capability: {:?}",
            session.reports()
        );
    }
}
