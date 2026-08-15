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
    // Round-2 finding B14b: only a definite `Absent` skips the build. An
    // `Unknown` — rustc unrunnable, a libdir that could not be read — is not
    // evidence that the target is missing, and reporting one as
    // `ToolchainMissing` is a green `inf cook-mods` with no wasm in it. When
    // the question cannot be asked, ask cargo instead by trying.
    if wasm_target() == WasmTarget::Absent {
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
        // C4-40: classify by **asking the toolchain**, not by reading the error
        // text. The old `target_missing(&stderr)` substring sniff matched
        // "target `wasm32-unknown-unknown`" — a phrase an ordinary compile error
        // for that target routinely contains — and reported a broken mod as
        // "toolchain missing", which `inf cook-mods` then exits 0 on. The probe
        // ran before the build too, so if it still says installed, the failure is
        // the code's.
        match wasm_target() {
            WasmTarget::Absent => {
                return Ok(ModBuildOutcome::ToolchainMissing(
                    TOOLCHAIN_INSTRUCTIONS.to_string(),
                ))
            }
            // B14b again, and this is the half that ships: an unreadable libdir
            // used to answer "absent" here, so a mod that genuinely does not
            // compile was re-classified as a missing toolchain and the cook
            // exited 0. It is a build failure, and the reason the probe could
            // not confirm otherwise travels with it.
            WasmTarget::Unknown(why) => {
                return Err(CookError::Mod(format!(
                    "mod build failed, and the wasm32-unknown-unknown target could \
                     not be verified ({why}):\n{stderr}"
                )))
            }
            WasmTarget::Present => {}
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

/// What this machine can say about the `wasm32-unknown-unknown` target's
/// **standard library** (round-2 finding **B14b**).
///
/// Three answers, not two. The boolean version collapsed *"the toolchain says
/// this target is not installed"* into the same value as *"this question could
/// not be asked"* — `rustc` unrunnable, a non-zero exit, a libdir that could
/// not be read — and every one of those made `build_mod_wasm` report a mod with
/// a genuine compile error as `ToolchainMissing`, which `inf cook-mods` exits
/// **0** on with no wasm produced. That is the silent-ship outcome C4-40
/// closed, reached through another door.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WasmTarget {
    /// The libdir exists and holds a `libstd` rlib.
    Present,
    /// The toolchain answered, and the standard library is not there.
    Absent,
    /// The question could not be asked. **Not** a synonym for `Absent`: a
    /// caller must not report a missing toolchain on this, because it has no
    /// evidence of one.
    Unknown(String),
}

/// Ask the toolchain directly.
///
/// # Why the exit status is not the answer
///
/// The first cut classified by reading `cargo`'s stderr for the target triple;
/// it also matched ordinary compile errors naming the target — and thereby
/// exposed the false positive underneath: on a machine without the target, the
/// build failed with `can't find crate for 'std'` and was reported as a **build
/// failure** rather than a missing toolchain. That is what turned the P14.5
/// `mods_e2e` skip into a panic on the CI legs that do not install wasm32.
///
/// Checking that the libdir exists **and holds a `libstd` rlib** asks the
/// toolchain the question directly, and keeps the property C4-40 was for: a
/// genuine compile error is still an error, because the target is still there.
pub fn wasm_target() -> WasmTarget {
    let out = match Command::new("rustc")
        .args([
            "--print",
            "target-libdir",
            "--target",
            "wasm32-unknown-unknown",
        ])
        .output()
    {
        Ok(out) => out,
        Err(e) => return WasmTarget::Unknown(format!("could not run rustc: {e}")),
    };
    if !out.status.success() {
        return WasmTarget::Unknown(format!(
            "rustc --print target-libdir exited {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    classify_libdir(&PathBuf::from(
        String::from_utf8_lossy(&out.stdout).trim().to_string(),
    ))
}

/// The verdict a target libdir carries — split out of [`wasm_target`] so the
/// three answers can be driven from a test (the rest of that function needs a
/// `rustc`, and the machine running the test already has the real one).
fn classify_libdir(dir: &Path) -> WasmTarget {
    // **The B14b line.** A libdir that cannot be READ is not a libdir that is
    // not there — a permission error, a half-unpacked toolchain, a busy network
    // share or a path that is a file all land here, and reporting them as "the
    // target is not installed" turns a broken environment into a green cook
    // with no wasm in it. `NotFound` is the one error that really does mean
    // absent.
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return WasmTarget::Absent,
        Err(e) => {
            return WasmTarget::Unknown(format!(
                "could not read the target libdir at {}: {e}",
                dir.display()
            ))
        }
    };
    // `flatten`, not `filter_map(Result::ok)`: this crate aliases `Result<T>` to
    // its own `CookError` result, which makes the path-qualified form ambiguous.
    if entries.flatten().any(|e| {
        e.file_name()
            .to_str()
            .is_some_and(|n| n.starts_with("libstd-") && n.ends_with(".rlib"))
    }) {
        WasmTarget::Present
    } else {
        WasmTarget::Absent
    }
}

/// [`wasm_target`] as a boolean, for the callers that genuinely want one.
///
/// **`Unknown` reads as `false` here**, so this is only safe where "not proven
/// present" is the conservative answer — never where the `false` becomes a
/// *report* that the toolchain is missing. See B14b.
pub fn wasm_target_installed() -> bool {
    wasm_target() == WasmTarget::Present
}

const TOOLCHAIN_INSTRUCTIONS: &str = "\
the wasm32-unknown-unknown target is not installed; mod wasm was NOT built.
Install it and re-run:
    rustup target add wasm32-unknown-unknown
    cargo build --release --target wasm32-unknown-unknown --manifest-path <mod>/Cargo.toml";

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

    /// **Round-2 finding B14b**: "the question could not be asked" is not "the
    /// answer is no".
    ///
    /// `wasm_target_installed` collapsed an unreadable libdir into `false`, so
    /// a mod with a genuine compile error was re-classified as
    /// `ToolchainMissing` and `inf cook-mods` exited **0 with no wasm** — the
    /// silent-ship outcome C4-40 closed, through another door.
    #[test]
    fn an_unreadable_libdir_is_not_an_absent_target() {
        let dir = tempfile::tempdir().unwrap();

        // Absent: nothing there at all.
        assert_eq!(
            classify_libdir(&dir.path().join("nope")),
            WasmTarget::Absent,
            "a libdir that does not exist really is absent"
        );

        // Absent: a real directory with no standard library in it.
        let empty = dir.path().join("empty");
        std::fs::create_dir(&empty).unwrap();
        std::fs::write(empty.join("libcore-1234.rlib"), b"").unwrap();
        assert_eq!(
            classify_libdir(&empty),
            WasmTarget::Absent,
            "a libdir with no libstd is absent"
        );

        // Present: the thing being looked for.
        let full = dir.path().join("full");
        std::fs::create_dir(&full).unwrap();
        std::fs::write(full.join("libstd-deadbeef.rlib"), b"").unwrap();
        assert_eq!(classify_libdir(&full), WasmTarget::Present);

        // UNKNOWN: the libdir path is a file. `read_dir` fails with something
        // that is not `NotFound` on every platform — which is the whole class
        // (a permission error, a half-unpacked toolchain, a network share) in
        // the one shape a test can produce portably.
        let file = dir.path().join("a-file");
        std::fs::write(&file, b"not a directory").unwrap();
        match classify_libdir(&file) {
            WasmTarget::Unknown(why) => assert!(
                why.contains("could not read"),
                "the reason must travel with the verdict: {why}"
            ),
            other => panic!("an unreadable libdir classified as {other:?}"),
        }

        // …and the boolean face reads Unknown as "not proven present", which is
        // only safe because no caller turns that `false` into a report.
        assert!(!matches!(classify_libdir(&file), WasmTarget::Present));
    }

    /// The two `build_mod_wasm` call sites act on `Absent` and never on
    /// `Unknown` — a source pin, because reaching either branch needs a machine
    /// whose libdir cannot be read.
    #[test]
    fn the_toolchain_missing_verdict_is_only_ever_reported_for_absent() {
        let src = include_str!("mods.rs").replace("\r\n", "\n");
        let body = {
            let at = src
                .find("pub fn build_mod_wasm(")
                .expect("`build_mod_wasm` occurs nowhere — was it renamed?");
            let rest = &src[at..];
            let end = rest.find("\n}\n").unwrap_or(rest.len());
            rest[..end].to_string()
        };
        assert_eq!(
            body.matches("ToolchainMissing").count(),
            3,
            "this pin is calibrated against `build_mod_wasm`'s two report sites plus the one mention in the comment that explains them"
        );
        assert!(
            !body.contains("!wasm_target_installed()"),
            "`build_mod_wasm` classifies with the boolean face again, which reads \
             `Unknown` as `false` and turns an unreadable libdir into a green cook \
             with no wasm in it (B14b)"
        );
        assert!(
            body.contains("wasm_target() == WasmTarget::Absent"),
            "`build_mod_wasm`'s pre-build probe no longer requires a DEFINITE absence"
        );
        assert!(
            body.contains("WasmTarget::Unknown(why)"),
            "`build_mod_wasm` no longer distinguishes an unanswerable probe from an \
             absent target when a build fails"
        );
    }
}
