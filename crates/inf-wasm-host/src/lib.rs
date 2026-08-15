//! Sandboxed WASM mod runtime (ROADMAP P14.5, deliverable 1).
//!
//! This is the **safe sibling** of the dylib hot-reload tier ([`inf-hotreload`]).
//! Where dylib plugins share a `#[repr(C)]` vtable and run with full process
//! trust (ABI-fragile, `unsafe`), a WASM mod runs inside `wasmtime`:
//!
//! * **memory-isolated** — it can only touch its own linear memory, capped by a
//!   store limiter;
//! * **capability-scoped** — only the host functions its [`ModCaps`] grant are
//!   linked, deny-by-default (an ungranted import fails instantiation with a
//!   clear message);
//! * **time-bounded** — a fuel (deterministic) or epoch (wall-clock) budget means
//!   a mod cannot hang the frame; a trapped/hung mod is disabled + reported and
//!   the host survives.
//!
//! The engine embedder implements one narrow trait — [`ModWorld`] — the exact
//! analogue of the blueprint interpreter's `Host`. The runtime/editor provide a
//! real one over the `inf-ecs` world; tests use [`MockModWorld`].
//!
//! Mods are authored in the *same* Blueprints/Rust and compiled blueprint → Rust
//! → wasm by the cook target (`inf-packager::mods`); the [`inf-mod`] shim wraps
//! this host ABI in safe Rust the generated (or hand-written) mod code calls.
//! "Two ways to code, one truth" — no new scripting language.
//!
//! [`inf-hotreload`]: ../inf_hotreload/index.html
//! [`inf-mod`]: ../inf_mod/index.html

mod caps;
mod host;
mod manifest;
mod world;

pub use caps::{Capability, ModCaps};
pub use host::{ExecLimits, ModError, ModTrap, WasmEngine, WasmMod, WasmMods};
pub use manifest::{ManifestError, ModManifest};
pub use world::{MockModWorld, ModWorld};

/// Assemble a WebAssembly-text module into a `.wasm` binary. A convenience for
/// tests (and tooling) that want a tiny real module without a wasm toolchain.
pub fn wat_to_wasm(src: &str) -> Result<Vec<u8>, ModError> {
    wat::parse_str(src).map_err(|e| ModError::Compile(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// A mod that moves entity 1 to (5,6,7) every update — needs `entities`.
    const MOVE_WAT: &str = r#"
        (module
          (import "env" "set_entity_translation"
            (func $set (param i64 f64 f64 f64)))
          (memory (export "memory") 1)
          (func (export "mod_update") (param $dt f64)
            (call $set (i64.const 1) (f64.const 5) (f64.const 6) (f64.const 7))))
    "#;

    /// A mod that logs "hi" — needs `log`.
    const LOG_WAT: &str = r#"
        (module
          (import "env" "log" (func $log (param i32 i32)))
          (memory (export "memory") 1)
          (data (i32.const 8) "hi")
          (func (export "mod_update") (param $dt f64)
            (call $log (i32.const 8) (i32.const 2))))
    "#;

    /// A mod with a **BeginPlay**: `mod_init` puts entity 1 at (1,2,3) and each
    /// `mod_update` moves it to (5,6,7). Reading (1,2,3) proves init ran; reading
    /// (5,6,7) after a tick proves init ran *before* it and only once.
    const INIT_WAT: &str = r#"
        (module
          (import "env" "set_entity_translation"
            (func $set (param i64 f64 f64 f64)))
          (memory (export "memory") 1)
          (func (export "mod_init")
            (call $set (i64.const 1) (f64.const 1) (f64.const 2) (f64.const 3)))
          (func (export "mod_update") (param $dt f64)
            (call $set (i64.const 1) (f64.const 5) (f64.const 6) (f64.const 7))))
    "#;

    /// A mod that hangs forever (a back-edge loop) — exhausts its budget.
    const HANG_WAT: &str = r#"
        (module
          (memory (export "memory") 1)
          (func (export "mod_update") (param $dt f64)
            (loop $l (br $l))))
    "#;

    fn engine() -> WasmEngine {
        WasmEngine::new().unwrap()
    }

    #[test]
    fn granted_capability_moves_entity() {
        let mut world = MockModWorld::with_entities([(1, [0.0, 0.0, 0.0])]);
        let mut m = WasmMod::instantiate(
            &engine(),
            "mover",
            MOVE_WAT.as_bytes(),
            ModCaps {
                entities: true,
                ..ModCaps::NONE
            },
            ExecLimits::default(),
        )
        .unwrap();
        m.update(&mut world, 0.016).unwrap();
        assert_eq!(world.entities[&1], [5.0, 6.0, 7.0]);
    }

    fn init_mod(engine: &WasmEngine) -> WasmMod {
        WasmMod::instantiate(
            engine,
            "beginner",
            INIT_WAT.as_bytes(),
            ModCaps {
                entities: true,
                ..ModCaps::NONE
            },
            ExecLimits::default(),
        )
        .unwrap()
    }

    /// A mod's `mod_init` is its BeginPlay. It was emitted by the cook target and
    /// documented by `inf_mod::infinity_mod!`, and the host only ever looked up
    /// `mod_update` — so the handler was compiled, shipped and **never called**.
    #[test]
    fn a_mods_begin_play_runs_once_before_its_first_tick() {
        let e = engine();
        let mut world = MockModWorld::with_entities([(1, [0.0, 0.0, 0.0])]);
        let mut m = init_mod(&e);
        assert!(m.has_init(), "the module exports mod_init");
        assert!(!m.initialized());

        // Explicitly, without any tick: BeginPlay's own writes land.
        assert!(m.init(&mut world).unwrap(), "init ran");
        assert_eq!(world.entities[&1], [1.0, 2.0, 3.0]);
        assert!(m.initialized());

        // Exactly once — a second call is a no-op even after the world moved.
        world.entities.insert(1, [9.0, 9.0, 9.0]);
        assert!(!m.init(&mut world).unwrap(), "init does not run twice");
        assert_eq!(world.entities[&1], [9.0, 9.0, 9.0]);
    }

    /// The one door: a host that only ticks still gets BeginPlay, before the
    /// first Tick. Ticking is how every production caller drives a mod
    /// (`WasmMods::tick`), so an `init` the tick path did not run would be a
    /// second path to forget.
    #[test]
    fn ticking_a_fresh_mod_runs_begin_play_first() {
        let e = engine();
        let mut world = MockModWorld::with_entities([(1, [0.0, 0.0, 0.0])]);
        let mut m = init_mod(&e);

        m.update(&mut world, 0.016).unwrap();
        // Tick ran after init, so the tick's value is what survives …
        assert_eq!(world.entities[&1], [5.0, 6.0, 7.0]);
        assert!(m.initialized(), "the tick ran init");

        // … and init does not fire again on the second tick: park the entity at
        // init's value and confirm the tick alone moves it.
        world.entities.insert(1, [1.0, 2.0, 3.0]);
        m.update(&mut world, 0.016).unwrap();
        assert_eq!(world.entities[&1], [5.0, 6.0, 7.0]);
    }

    /// A mod with only a Tick handler exports no `mod_init`, and that is not an
    /// error — the export is genuinely optional.
    #[test]
    fn a_mod_without_begin_play_still_loads_and_ticks() {
        let e = engine();
        let mut world = MockModWorld::with_entities([(1, [0.0, 0.0, 0.0])]);
        let mut m = WasmMod::instantiate(
            &e,
            "mover",
            MOVE_WAT.as_bytes(),
            ModCaps {
                entities: true,
                ..ModCaps::NONE
            },
            ExecLimits::default(),
        )
        .unwrap();
        assert!(!m.has_init());
        assert!(!m.init(&mut world).unwrap(), "nothing to run");
        m.update(&mut world, 0.016).unwrap();
        assert_eq!(world.entities[&1], [5.0, 6.0, 7.0]);
    }

    #[test]
    fn ungranted_capability_fails_to_instantiate_with_clear_message() {
        // Same module, but WITHOUT the `entities` cap.
        let err = WasmMod::instantiate(
            &engine(),
            "mover",
            MOVE_WAT.as_bytes(),
            ModCaps::NONE,
            ExecLimits::default(),
        )
        .err()
        .unwrap();
        match err {
            ModError::MissingCapability { ref import, cap } => {
                assert_eq!(import, "set_entity_translation");
                assert_eq!(cap, Capability::Entities);
            }
            other => panic!("expected MissingCapability, got {other}"),
        }
        // And the message names the capability.
        assert!(err.to_string().contains("entities"), "{err}");
    }

    #[test]
    fn unknown_import_is_rejected() {
        let wat = r#"
            (module
              (import "env" "definitely_not_a_host_fn" (func))
              (memory (export "memory") 1)
              (func (export "mod_update") (param f64)))
        "#;
        let err = WasmMod::instantiate(
            &engine(),
            "bad",
            wat.as_bytes(),
            ModCaps::ALL,
            ExecLimits::default(),
        )
        .err()
        .unwrap();
        assert!(matches!(err, ModError::UnknownImport(_)), "{err}");
    }

    #[test]
    fn log_capability_routes_to_world() {
        let mut world = MockModWorld::default();
        let mut m = WasmMod::instantiate(
            &engine(),
            "logger",
            LOG_WAT.as_bytes(),
            ModCaps {
                log: true,
                ..ModCaps::NONE
            },
            ExecLimits::default(),
        )
        .unwrap();
        m.update(&mut world, 0.016).unwrap();
        assert_eq!(world.logs, vec!["hi".to_string()]);
    }

    #[test]
    fn hung_mod_traps_on_fuel_and_host_survives() {
        let mut world = MockModWorld::default();
        // A small fuel budget so the infinite loop trips fast + deterministically.
        let limits = ExecLimits {
            fuel_per_update: Some(1_000_000),
            ..ExecLimits::default()
        };
        let mut hung = WasmMod::instantiate(
            &engine(),
            "hung",
            HANG_WAT.as_bytes(),
            ModCaps::NONE,
            limits,
        )
        .unwrap();
        let trap = hung.update(&mut world, 0.016).unwrap_err();
        assert!(matches!(trap, ModTrap::Trap(_)), "{trap}");
        assert!(!hung.enabled(), "a trapped mod must be disabled");
        // Ticking it again is a no-op — the host is unaffected.
        assert_eq!(hung.update(&mut world, 0.016), Err(ModTrap::Disabled));

        // A *good* mod on the SAME engine still works — the host survived.
        let mut good = WasmMod::instantiate(
            &engine(),
            "mover",
            MOVE_WAT.as_bytes(),
            ModCaps {
                entities: true,
                ..ModCaps::NONE
            },
            ExecLimits::default(),
        )
        .unwrap();
        let mut w2 = MockModWorld::with_entities([(1, [0.0; 3])]);
        good.update(&mut w2, 0.016).unwrap();
        assert_eq!(w2.entities[&1], [5.0, 6.0, 7.0]);
    }

    #[test]
    fn hung_mod_traps_on_wall_clock_epoch() {
        // A wall-clock engine: the epoch advances every 1ms; a 5-tick deadline
        // (~5ms) with fuel DISABLED means only the timer stops the infinite loop.
        let engine = WasmEngine::with_epoch(Duration::from_millis(1)).unwrap();
        let limits = ExecLimits {
            fuel_per_update: None,
            epoch_deadline: Some(5),
            ..ExecLimits::default()
        };
        let mut hung =
            WasmMod::instantiate(&engine, "hung", HANG_WAT.as_bytes(), ModCaps::NONE, limits)
                .unwrap();
        let mut world = MockModWorld::default();
        let trap = hung.update(&mut world, 0.016).unwrap_err();
        assert!(matches!(trap, ModTrap::Trap(_)), "{trap}");
        assert!(!hung.enabled());
    }

    #[test]
    fn spawn_and_input_capabilities() {
        // Spawns a cube, then moves entity 1 if "boost" is held.
        let wat = r#"
            (module
              (import "env" "spawn_cube" (func $spawn (param f64 f64 f64) (result i64)))
              (import "env" "input_is_down" (func $down (param i32 i32) (result i32)))
              (import "env" "set_entity_translation" (func $set (param i64 f64 f64 f64)))
              (memory (export "memory") 1)
              (data (i32.const 0) "boost")
              (func (export "mod_update") (param $dt f64)
                (drop (call $spawn (f64.const 1) (f64.const 2) (f64.const 3)))
                (if (call $down (i32.const 0) (i32.const 5))
                  (then (call $set (i64.const 1) (f64.const 9) (f64.const 9) (f64.const 9))))))
        "#;
        let caps = ModCaps {
            spawn: true,
            input: true,
            entities: true,
            log: false,
        };
        let mut m =
            WasmMod::instantiate(&engine(), "s", wat.as_bytes(), caps, ExecLimits::default())
                .unwrap();

        // Without "boost": spawns a cube, entity 1 unchanged.
        let mut world = MockModWorld::with_entities([(1, [0.0; 3])]);
        m.update(&mut world, 0.016).unwrap();
        assert_eq!(world.spawned.len(), 1);
        assert_eq!(world.entities[&1], [0.0; 3]);

        // With "boost" held: entity 1 jumps to (9,9,9).
        let mut world2 = MockModWorld::with_entities([(1, [0.0; 3])]).press("boost");
        let mut m2 =
            WasmMod::instantiate(&engine(), "s", wat.as_bytes(), caps, ExecLimits::default())
                .unwrap();
        m2.update(&mut world2, 0.016).unwrap();
        assert_eq!(world2.entities[&1], [9.0, 9.0, 9.0]);
    }

    #[test]
    fn entity_translation_out_param_roundtrips() {
        // Reads entity 1's translation into a scratch buffer, then writes it to
        // entity 2 — proving the out-param read path.
        let wat = r#"
            (module
              (import "env" "entity_translation" (func $get (param i64 i32) (result i32)))
              (import "env" "set_entity_translation" (func $set (param i64 f64 f64 f64)))
              (memory (export "memory") 1)
              (func (export "mod_update") (param $dt f64)
                (drop (call $get (i64.const 1) (i32.const 16)))
                (call $set (i64.const 2)
                  (f64.load (i32.const 16))
                  (f64.load (i32.const 24))
                  (f64.load (i32.const 32)))))
        "#;
        let mut world = MockModWorld::with_entities([(1, [1.5, 2.5, 3.5]), (2, [0.0, 0.0, 0.0])]);
        let mut m = WasmMod::instantiate(
            &engine(),
            "copy",
            wat.as_bytes(),
            ModCaps {
                entities: true,
                ..ModCaps::NONE
            },
            ExecLimits::default(),
        )
        .unwrap();
        m.update(&mut world, 0.016).unwrap();
        assert_eq!(world.entities[&2], [1.5, 2.5, 3.5]);
    }

    #[test]
    fn missing_update_export_is_rejected() {
        let wat = r#"(module (memory (export "memory") 1))"#;
        let err = WasmMod::instantiate(
            &engine(),
            "x",
            wat.as_bytes(),
            ModCaps::NONE,
            ExecLimits::default(),
        )
        .err()
        .unwrap();
        assert!(
            matches!(err, ModError::MissingExport("mod_update")),
            "{err}"
        );
    }

    #[test]
    fn wat_helper_produces_loadable_binary() {
        let bytes = wat_to_wasm(MOVE_WAT).unwrap();
        assert_eq!(&bytes[0..4], b"\0asm");
        let mut world = MockModWorld::with_entities([(1, [0.0; 3])]);
        let mut m = WasmMod::instantiate(
            &engine(),
            "mover",
            &bytes,
            ModCaps {
                entities: true,
                ..ModCaps::NONE
            },
            ExecLimits::default(),
        )
        .unwrap();
        m.update(&mut world, 0.016).unwrap();
        assert_eq!(world.entities[&1], [5.0, 6.0, 7.0]);
    }
}
