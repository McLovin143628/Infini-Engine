//! End-to-end proof (P14.5, deliverable 2): compile the committed `samples/mods/
//! spinner` Rust mod to `wasm32-unknown-unknown`, load it through the real
//! sandbox with its `mod.toml` capability grant, and tick it against a mock world
//! — asserting it orbits its entity.
//!
//! Skips (does not fail) when the `wasm32-unknown-unknown` target is not
//! installed. CI installs it (the wasm-check job), so there it runs for real.

use std::path::{Path, PathBuf};
use std::process::Command;

use inf_wasm_host::{ExecLimits, MockModWorld, ModCaps, ModManifest, WasmEngine, WasmMod};

/// Repo-root-relative path to the spinner sample.
fn spinner_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR = <repo>/crates/inf-wasm-host
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("samples")
        .join("mods")
        .join("spinner")
}

/// Build the spinner to wasm; `Ok(None)` means the target is not installed (skip).
fn build_spinner() -> Result<Option<PathBuf>, String> {
    let dir = spinner_dir();
    let manifest = dir.join("Cargo.toml");
    let output = Command::new("cargo")
        .args([
            "build",
            "--release",
            "--target",
            "wasm32-unknown-unknown",
            "--manifest-path",
        ])
        .arg(&manifest)
        // The nested build must not inherit `CARGO_TARGET_DIR` — the `.wasm` is
        // looked up under the crate's own `target/`, and an inherited override
        // (an isolated worktree, a shared CI cache, a developer's export) puts it
        // somewhere this test then reports as a missing toolchain. Same fix, same
        // reason, as `inf_packager::mods::build_mod_wasm`.
        .env_remove("CARGO_TARGET_DIR")
        .output()
        .map_err(|e| format!("spawning cargo: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        // Distinguish "target not installed" (skip) from a real build failure.
        if stderr.contains("may not be installed")
            || stderr.contains("target `wasm32-unknown-unknown`")
            || stderr.contains("std` is required")
        {
            return Ok(None);
        }
        return Err(format!("spinner build failed:\n{stderr}"));
    }

    let wasm = dir
        .join("target")
        .join("wasm32-unknown-unknown")
        .join("release")
        .join("spinner_mod.wasm");
    if !wasm.exists() {
        return Err(format!("expected wasm at {}", wasm.display()));
    }
    Ok(Some(wasm))
}

#[test]
fn spinner_mod_orbits_its_entity() {
    let wasm_path = match build_spinner() {
        Ok(Some(p)) => p,
        Ok(None) => {
            eprintln!("SKIP: wasm32-unknown-unknown target not installed");
            return;
        }
        Err(e) => panic!("{e}"),
    };

    // Load the mod's real capability grant from its committed mod.toml.
    let caps =
        ModManifest::parse(&std::fs::read_to_string(spinner_dir().join("mod.toml")).unwrap())
            .unwrap()
            .caps;
    assert_eq!(
        caps,
        ModCaps {
            entities: true,
            input: true,
            log: true,
            spawn: false,
        }
    );

    let wasm = std::fs::read(&wasm_path).unwrap();
    let engine = WasmEngine::new().unwrap();
    let mut m = WasmMod::instantiate(&engine, "spinner", &wasm, caps, ExecLimits::default())
        .expect("spinner instantiates");

    // Entity 1 starts at the origin; step a quarter-ish of the orbit.
    let mut world = MockModWorld::with_entities([(1, [0.0, 0.0, 0.0])]);
    let dt = 1.0 / 60.0;
    for _ in 0..30 {
        m.update(&mut world, dt).expect("spinner ticks");
    }

    let p = world.entities[&1];
    // y is pinned to 1.0 by the mod; x/z lie on the radius-2 circle and have
    // moved off the origin.
    assert!((p[1] - 1.0).abs() < 1e-9, "y pinned, got {p:?}");
    let r = (p[0] * p[0] + p[2] * p[2]).sqrt();
    assert!((r - 2.0).abs() < 1e-6, "on the orbit radius, got r={r}");
    assert!(p[0].abs() > 1e-6 || p[2].abs() > 1e-6, "moved off origin");
}
