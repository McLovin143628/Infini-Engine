//! The WASM mod cook target (ROADMAP P14.5, deliverable 2).
//!
//! A mod is authored in the *same* Blueprints/Rust as everything else — no new
//! scripting language. This module is the "→ wasm" leg of that pipeline: given a
//! blueprint [`BlueprintClass`], the **existing** transpiler
//! ([`inf_transpile::generate_fn`]) renders its event handlers to Rust, and we
//! wrap that Rust in a generated `cdylib` crate (crate-type `cdylib`, target
//! `wasm32-unknown-unknown`) with a thin **host shim** mapping the generated
//! code's host calls onto the [`inf-mod`] guest ABI + a `mod_update` entry the
//! [`inf-wasm-host`] sandbox invokes.
//!
//! # Scope (honest)
//!
//! v1 lowers the **mod host namespace** a mod blueprint targets — `host.*`
//! (entity transforms / spawn / log) and `input.*` — onto the sandbox ABI. The
//! generated crate compiles when the blueprint uses that subset. Bridging the
//! *entire* engine node kit (physics, audio, `vars`, arbitrary `engine::*`) onto
//! wasm imports is the documented follow-up; the committed hand-written
//! `samples/mods/spinner` proves the full author→wasm→sandbox path end to end
//! today. The generated template here is verified at the string level (the shim,
//! the entry, the transpiled body, the manifest).
//!
//! [`inf-mod`]: ../../inf_mod/index.html
//! [`inf-wasm-host`]: ../../inf_wasm_host/index.html

use std::path::{Path, PathBuf};
use std::process::Command;

use inf_blueprint::{BlueprintClass, EventKind};

use crate::error::{CookError, Result};

/// The generated source of a mod `cdylib` crate (in-memory; write with
/// [`GeneratedModCrate::write_to`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedModCrate {
    /// The crate name (sanitized from the class name).
    pub crate_name: String,
    /// `Cargo.toml` contents.
    pub cargo_toml: String,
    /// `src/lib.rs` contents (shim + transpiled handlers + entry points).
    pub lib_rs: String,
    /// `mod.toml` capability manifest contents.
    pub mod_toml: String,
}

impl GeneratedModCrate {
    /// Write the crate to `dir` (creating `dir` and `dir/src`).
    pub fn write_to(&self, dir: &Path) -> Result<()> {
        let src = dir.join("src");
        std::fs::create_dir_all(&src).map_err(CookError::from)?;
        std::fs::write(dir.join("Cargo.toml"), &self.cargo_toml).map_err(CookError::from)?;
        std::fs::write(dir.join("mod.toml"), &self.mod_toml).map_err(CookError::from)?;
        std::fs::write(src.join("lib.rs"), &self.lib_rs).map_err(CookError::from)?;
        Ok(())
    }
}

/// Options for generating a mod crate.
#[derive(Debug, Clone)]
pub struct ModBuildOptions {
    /// Relative path (from the generated crate) to the `inf-mod` shim crate. The
    /// default assumes the crate is written under `<repo>/.../<name>` two levels
    /// deep; callers writing elsewhere override it.
    pub inf_mod_path: String,
}

impl Default for ModBuildOptions {
    fn default() -> Self {
        Self {
            inf_mod_path: "../../crates/inf-mod".to_string(),
        }
    }
}

/// Lower a blueprint class into a generated mod crate: transpile its event
/// handlers with the shared transpiler, wrap them in the host shim + entry
/// points, and emit `Cargo.toml` + `mod.toml`.
pub fn generate_mod_crate(
    class: &BlueprintClass,
    opts: &ModBuildOptions,
) -> Result<GeneratedModCrate> {
    let crate_name = sanitize_crate_name(&class.name);

    // Transpile each event handler through the SAME codegen the compiled path
    // uses (parity), stripping the engine-only `#[infinity::blueprint]` marker
    // attribute (the mod crate has no such proc-macro).
    let mut handlers = String::new();
    let mut tick_fn: Option<String> = None;
    let mut begin_fn: Option<String> = None;
    for binding in &class.events {
        let rust = inf_transpile::generate_fn(&binding.body)
            .map_err(|e| CookError::Mod(format!("transpiling {}: {e}", binding.event.key())))?;
        handlers.push_str(&strip_blueprint_marker(&rust));
        handlers.push('\n');
        match binding.event {
            EventKind::Tick => tick_fn = Some(binding.body.name.clone()),
            EventKind::BeginPlay => begin_fn = Some(binding.body.name.clone()),
            _ => {}
        }
    }

    let lib_rs = render_lib_rs(
        &class.name,
        &handlers,
        tick_fn.as_deref(),
        begin_fn.as_deref(),
    );
    let cargo_toml = render_cargo_toml(&crate_name, &opts.inf_mod_path);
    let mod_toml = render_mod_toml(&class.name);

    Ok(GeneratedModCrate {
        crate_name,
        cargo_toml,
        lib_rs,
        mod_toml,
    })
}

/// The outcome of trying to build a mod crate to wasm.
#[derive(Debug)]
pub enum ModBuildOutcome {
    /// The `.wasm` was produced at this path.
    Built(PathBuf),
    /// The `wasm32-unknown-unknown` toolchain/target is missing — build skipped
    /// with honest instructions.
    ToolchainMissing(String),
}

/// Build a mod crate (a directory holding `Cargo.toml`) to
/// `wasm32-unknown-unknown` when the toolchain is present, else return
/// [`ModBuildOutcome::ToolchainMissing`] with instructions.
pub fn build_mod_wasm(crate_dir: &Path) -> Result<ModBuildOutcome> {
    if !wasm_target_installed() {
        return Ok(ModBuildOutcome::ToolchainMissing(
            TOOLCHAIN_INSTRUCTIONS.to_string(),
        ));
    }
    let manifest = crate_dir.join("Cargo.toml");
    let output = Command::new("cargo")
        .args([
            "build",
            "--release",
            "--target",
            "wasm32-unknown-unknown",
            "--manifest-path",
        ])
        .arg(&manifest)
        // **The nested build must not inherit `CARGO_TARGET_DIR`.** The artifact
        // is looked up under `<crate>/target/wasm32-unknown-unknown/release`, so
        // an inherited override — which is exactly what an isolated worktree, a
        // shared-target CI cache or a developer's own export sets — puts the
        // `.wasm` somewhere this function then reports as missing. Cleared rather
        // than honoured, because the path below is the contract.
        .env_remove("CARGO_TARGET_DIR")
        .output()
        .map_err(|e| CookError::Mod(format!("spawning cargo: {e}")))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if target_missing(&stderr) {
            return Ok(ModBuildOutcome::ToolchainMissing(
                TOOLCHAIN_INSTRUCTIONS.to_string(),
            ));
        }
        return Err(CookError::Mod(format!("mod build failed:\n{stderr}")));
    }
    // Locate the produced .wasm (the crate name with `-`→`_`).
    let dir = crate_dir
        .join("target")
        .join("wasm32-unknown-unknown")
        .join("release");
    let wasm = std::fs::read_dir(&dir)
        .map_err(CookError::from)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.extension().and_then(|s| s.to_str()) == Some("wasm"))
        .ok_or_else(|| CookError::Mod(format!("no .wasm produced in {}", dir.display())))?;
    Ok(ModBuildOutcome::Built(wasm))
}

/// Whether the `wasm32-unknown-unknown` target is installed for the active
/// toolchain.
pub fn wasm_target_installed() -> bool {
    Command::new("rustc")
        .args([
            "--print",
            "target-libdir",
            "--target",
            "wasm32-unknown-unknown",
        ])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

const TOOLCHAIN_INSTRUCTIONS: &str = "\
the wasm32-unknown-unknown target is not installed; mod wasm was NOT built.
Install it and re-run:
    rustup target add wasm32-unknown-unknown
    cargo build --release --target wasm32-unknown-unknown --manifest-path <mod>/Cargo.toml";

fn target_missing(stderr: &str) -> bool {
    stderr.contains("may not be installed")
        || stderr.contains("target `wasm32-unknown-unknown`")
        || stderr.contains("std` is required")
}

/// Strip the `#[infinity::blueprint(id = "…")]` marker attribute line(s) from
/// transpiled Rust — that proc-macro ships with the engine runtime, not a mod.
fn strip_blueprint_marker(rust: &str) -> String {
    rust.lines()
        .filter(|l| !l.trim_start().starts_with("#[infinity::blueprint"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Sanitize a class display name into a crate name (`a-z0-9-`).
fn sanitize_crate_name(name: &str) -> String {
    let mut out: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect();
    while out.contains("--") {
        out = out.replace("--", "-");
    }
    let trimmed = out.trim_matches('-').to_string();
    let base = if trimmed.is_empty() {
        "mod".to_string()
    } else {
        trimmed
    };
    format!("{base}-mod")
}

fn render_cargo_toml(crate_name: &str, inf_mod_path: &str) -> String {
    format!(
        "# GENERATED by `inf cook --mods` (P14.5). Edit the blueprint, not this file.\n\
         [package]\n\
         name = \"{crate_name}\"\n\
         version = \"0.0.0\"\n\
         edition = \"2021\"\n\
         publish = false\n\n\
         [lib]\n\
         crate-type = [\"cdylib\"]\n\n\
         [dependencies]\n\
         inf-mod = {{ path = \"{inf_mod_path}\" }}\n\n\
         [profile.release]\n\
         opt-level = \"s\"\n\
         lto = true\n\n\
         # Standalone workspace: this crate builds only for wasm32.\n\
         [workspace]\n"
    )
}

fn render_mod_toml(class_name: &str) -> String {
    // Conservative default grant: entities + log. An author widens this to match
    // the blueprint's host calls (the cook could infer it — a documented
    // follow-up).
    format!(
        "# GENERATED capability grant for the `{class_name}` mod (P14.5).\n\
         # Deny-by-default: widen these to match the blueprint's host calls.\n\
         name = \"{class_name}\"\n\n\
         [caps]\n\
         entities = true\n\
         input = false\n\
         log = true\n\
         spawn = false\n"
    )
}

fn render_lib_rs(
    class_name: &str,
    handlers: &str,
    tick_fn: Option<&str>,
    begin_fn: Option<&str>,
) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "// GENERATED mod crate for blueprint class `{class_name}` (P14.5).\n\
         // Authored as a Blueprint; transpiled by inf-transpile (the SAME codegen\n\
         // the compiled/dylib path uses) and wrapped in the host shim below.\n\n\
         #![allow(clippy::all, unused)]\n\n"
    ));

    out.push_str(HOST_SHIM);
    out.push('\n');
    out.push_str("// ── transpiled blueprint event handlers ──\n");
    out.push_str(handlers);
    out.push('\n');

    out.push_str("// ── sandbox entry points (called by inf-wasm-host) ──\n");
    match tick_fn {
        Some(name) => out.push_str(&format!(
            "#[no_mangle]\npub extern \"C\" fn mod_update(dt: f64) {{ {name}(dt); }}\n"
        )),
        None => out.push_str(
            "#[no_mangle]\npub extern \"C\" fn mod_update(_dt: f64) {{ /* no Tick handler */ }}\n",
        ),
    }
    if let Some(name) = begin_fn {
        out.push_str(&format!(
            "#[no_mangle]\npub extern \"C\" fn mod_init() {{ {name}(); }}\n"
        ));
    }
    out
}

/// The generated host shim: maps the blueprint mod-host namespace onto the
/// `inf-mod` guest ABI. Blueprint calls like `host::set_entity_translation(..)`
/// and `input::is_down(..)` resolve here.
const HOST_SHIM: &str = "\
// ── generated host shim: blueprint host calls → the inf-mod guest ABI ──
#[allow(dead_code)]
mod host {
    pub fn set_entity_translation(entity: i64, x: f64, y: f64, z: f64) {
        inf_mod::set_translation(entity, [x, y, z]);
    }
    pub fn log(message: &str) {
        inf_mod::log(message);
    }
    pub fn spawn_cube(x: f64, y: f64, z: f64) -> i64 {
        inf_mod::spawn_cube(x, y, z)
    }
}
#[allow(dead_code)]
mod input {
    pub fn is_down(key: &str) -> bool {
        inf_mod::input_is_down(key)
    }
}
";

#[cfg(test)]
mod tests {
    use super::*;
    use inf_blueprint::{BinOp, BlueprintFn, EventBinding, Expr, Lit, Param, Stmt, Ty};

    /// A tiny "orbit" mod class: on Tick, read `dt`, and call
    /// `host::set_entity_translation(1, dt, 1.0, dt)`.
    fn orbit_class() -> BlueprintClass {
        let body = BlueprintFn {
            id: "tick".into(),
            name: "tick".into(),
            params: vec![Param {
                name: "dt".into(),
                ty: Ty::Float,
            }],
            ret: Ty::Unit,
            body: vec![Stmt::ExprStmt(Expr::Call {
                path: vec!["host".into(), "set_entity_translation".into()],
                args: vec![
                    Expr::Lit(Lit::Int(1)),
                    Expr::Param("dt".into()),
                    Expr::Lit(Lit::Float(1.0)),
                    Expr::Binary(
                        BinOp::Mul,
                        Box::new(Expr::Param("dt".into())),
                        Box::new(Expr::Lit(Lit::Float(2.0))),
                    ),
                ],
            })],
        };
        let mut class = BlueprintClass::new("act:orbit", "Orbit");
        class.events = vec![EventBinding {
            event: EventKind::Tick,
            body,
        }];
        class
    }

    #[test]
    fn generates_cdylib_crate_with_shim_and_entry() {
        let g = generate_mod_crate(&orbit_class(), &ModBuildOptions::default()).unwrap();

        assert_eq!(g.crate_name, "orbit-mod");
        // Cargo.toml is a wasm cdylib depending on the shim.
        assert!(g.cargo_toml.contains("crate-type = [\"cdylib\"]"));
        assert!(g.cargo_toml.contains("inf-mod = { path ="));
        assert!(g.cargo_toml.contains("[workspace]"));
        // lib.rs carries the host shim, the transpiled body, and the entry.
        assert!(g.lib_rs.contains("mod host"));
        assert!(g.lib_rs.contains("inf_mod::set_translation"));
        assert!(
            g.lib_rs.contains("set_entity_translation"),
            "transpiled call present"
        );
        assert!(g.lib_rs.contains("fn tick(dt: f64)"), "{}", g.lib_rs);
        assert!(g.lib_rs.contains("pub extern \"C\" fn mod_update"));
        // The engine-only marker attribute is stripped.
        assert!(!g.lib_rs.contains("#[infinity::blueprint"));
        // mod.toml is deny-by-default-ish (spawn off).
        assert!(g.mod_toml.contains("spawn = false"));
    }

    #[test]
    fn writes_crate_to_disk() {
        let g = generate_mod_crate(&orbit_class(), &ModBuildOptions::default()).unwrap();
        let dir = tempfile::tempdir().unwrap();
        g.write_to(dir.path()).unwrap();
        assert!(dir.path().join("Cargo.toml").exists());
        assert!(dir.path().join("src/lib.rs").exists());
        assert!(dir.path().join("mod.toml").exists());
    }

    #[test]
    fn sanitizes_odd_class_names() {
        assert_eq!(sanitize_crate_name("My Cool Actor!"), "my-cool-actor-mod");
        assert_eq!(sanitize_crate_name("___"), "mod-mod");
    }
}
