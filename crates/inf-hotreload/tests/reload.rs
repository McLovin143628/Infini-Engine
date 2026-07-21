//! The real thing: build the two fixture plugin cdylibs with cargo, load
//! them through the shadow-copy path, and prove the Spike C exit criteria —
//! state carried across a reload, schema migration defaulting new fields,
//! panic containment with reload re-enabling, and the Windows file-lock
//! dance (the original artifact stays writable while its copy is loaded).

use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use inf_hotreload::{HotWorld, InstanceStatus, PluginHost};

const DT: f64 = 1.0 / 60.0;

/// Build both fixture plugins (once per test process) and return their
/// artifact paths, parsed from cargo's JSON messages so `CARGO_TARGET_DIR`
/// overrides and per-OS dylib naming are handled for us.
fn fixture_dylibs() -> &'static (PathBuf, PathBuf) {
    static BUILT: OnceLock<(PathBuf, PathBuf)> = OnceLock::new();
    BUILT.get_or_init(|| {
        let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
        let output = Command::new(cargo)
            .current_dir(env!("CARGO_MANIFEST_DIR"))
            .args([
                "build",
                "-p",
                "inf-hotreload-test-plugin-v1",
                "-p",
                "inf-hotreload-test-plugin-v2",
                "--message-format=json",
            ])
            .output()
            .expect("failed to run cargo build for fixture plugins");
        assert!(
            output.status.success(),
            "fixture plugin build failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let mut v1 = None;
        let mut v2 = None;
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            let Ok(msg) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            if msg["reason"] != "compiler-artifact" {
                continue;
            }
            let slot = match msg["target"]["name"].as_str().unwrap_or("") {
                "inf-hotreload-test-plugin-v1" | "inf_hotreload_test_plugin_v1" => &mut v1,
                "inf-hotreload-test-plugin-v2" | "inf_hotreload_test_plugin_v2" => &mut v2,
                _ => continue,
            };
            for name in msg["filenames"].as_array().into_iter().flatten() {
                let path = PathBuf::from(name.as_str().unwrap_or(""));
                if path
                    .extension()
                    .is_some_and(|e| e == std::env::consts::DLL_EXTENSION)
                {
                    *slot = Some(path);
                }
            }
        }
        // Copy the built artifacts into a process-private directory and load
        // from those copies. nextest runs each test in its own process, so
        // several of these test binaries invoke `cargo build` concurrently and
        // may re-touch the shared `target/` dylibs; loading from a private copy
        // makes the fixtures immune to a rebuild tearing the file mid-read
        // (an intermittent macOS-only `dlopen` failure otherwise).
        let stash = std::env::temp_dir().join(format!("inf-hotreload-fix-{}", std::process::id()));
        std::fs::create_dir_all(&stash).expect("create fixture stash");
        let stable = |src: PathBuf| -> PathBuf {
            let dst = stash.join(src.file_name().expect("dylib file name"));
            std::fs::copy(&src, &dst).expect("copy fixture dylib");
            dst
        };
        (
            stable(v1.expect("no v1 dylib in cargo output")),
            stable(v2.expect("no v2 dylib in cargo output")),
        )
    })
}

fn shadow_dir() -> PathBuf {
    // Shadow copies stay loaded (and on Windows therefore locked) until the
    // test process exits, so no cleanup — the OS temp dir owns them.
    std::env::temp_dir().join(format!("inf-hotreload-spike-{}", std::process::id()))
}

#[test]
fn full_reload_story() {
    let (v1_path, v2_path) = fixture_dylibs();
    let host = PluginHost::new(shadow_dir()).unwrap();

    let v1 = host.load(v1_path).unwrap();
    assert_eq!(v1.version(), "1.0.0");
    let mut names: Vec<_> = v1.component_names().collect();
    names.sort_unstable();
    assert_eq!(names, ["counter", "fragile"]);

    let mut world = HotWorld::new();
    let counter = world.spawn(&v1, "counter").unwrap();
    let fragile = world.spawn(&v1, "fragile").unwrap();

    for _ in 0..5 {
        world.tick(DT);
    }

    // v1 behavior: counter +1/tick; fragile panicked at count 3 and was
    // disabled without harming the process or the counter.
    assert_eq!(world.state_json(counter).unwrap()["ticks"], 5);
    assert_eq!(world.status(fragile), Some(InstanceStatus::Disabled));
    let error = world.last_error(fragile).unwrap();
    assert!(error.contains("fragile v1 exploded"), "got: {error}");
    assert_eq!(world.state_json(fragile).unwrap()["count"], 3);
    assert!(world
        .logs()
        .iter()
        .any(|r| r.message.contains("counter v1: first tick")));

    // Reload onto v2: counter's schema changed (migrated, bonus defaulted),
    // fragile's didn't (carried) and the reload re-enables it.
    let v2 = host.load(v2_path).unwrap();
    assert_eq!(v2.version(), "2.0.0");
    let report = world.reload(&v2);
    assert_eq!(report.migrated, vec![counter]);
    assert_eq!(report.carried, vec![fragile]);
    assert!(report.removed.is_empty(), "{:?}", report.removed);
    assert!(report.failed.is_empty(), "{:?}", report.failed);
    assert_eq!(world.status(fragile), Some(InstanceStatus::Running));

    let migrated = world.state_json(counter).unwrap();
    assert_eq!(migrated["ticks"], 5, "state must survive the reload");
    assert_eq!(migrated["bonus"], 1.5, "new field must take its default");

    // v2 behavior takes over on the carried state.
    world.tick(DT);
    assert_eq!(world.state_json(counter).unwrap()["ticks"], 15);
    assert_eq!(world.state_json(fragile).unwrap()["count"], 4);
    assert_eq!(world.status(fragile), Some(InstanceStatus::Running));
}

#[test]
fn loading_is_content_addressed() {
    let (v1_path, _) = fixture_dylibs();
    let host = PluginHost::new(shadow_dir()).unwrap();
    let a = host.load(v1_path).unwrap();
    let b = host.load(v1_path).unwrap();
    assert_eq!(a.content_hash(), b.content_hash());
    assert_eq!(a.shadow_path(), b.shadow_path());
}

/// The Windows file-lock dance: because the host loads a shadow copy, the
/// *original* dylib stays writable — the next `cargo build` (simulated here
/// by overwriting it with different bytes) is never blocked, and reloading
/// the changed file picks up the new code under a new shadow path.
#[test]
fn original_dylib_stays_writable_while_loaded() {
    let (v1_path, v2_path) = fixture_dylibs();
    let scratch = shadow_dir().join("build-output");
    std::fs::create_dir_all(&scratch).unwrap();
    let live = scratch.join(format!("live_plugin.{}", std::env::consts::DLL_EXTENSION));
    std::fs::copy(v1_path, &live).unwrap();

    let host = PluginHost::new(shadow_dir()).unwrap();
    let old = host.load(&live).unwrap();
    assert_eq!(old.version(), "1.0.0");

    // Would fail with ERROR_SHARING_VIOLATION on Windows if `live` itself
    // were loaded.
    std::fs::copy(v2_path, &live).expect("original must remain writable while its copy is loaded");

    let new = host.load(&live).unwrap();
    assert_eq!(new.version(), "2.0.0");
    assert_ne!(old.content_hash(), new.content_hash());
    assert_ne!(old.shadow_path(), new.shadow_path());
}
