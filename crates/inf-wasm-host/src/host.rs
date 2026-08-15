//! The sandbox: a wasmtime engine, capability-scoped linker, and the loaded-mod
//! runtime (P14.5, deliverable 1).
//!
//! # The host ABI (v1)
//!
//! A mod is a plain core wasm module behind a narrow `extern "C"`-style ABI of
//! `i64`/`f64`/`i32` — no interface types, no component model. It **imports**
//! (module `"env"`) the host functions its capabilities unlock, and **exports**:
//!
//! * `memory` — its linear memory (required).
//! * `mod_update(dt: f64)` — ticked once per fixed step (required).
//! * `mod_init()` — the mod's **BeginPlay**, called once immediately before its
//!   first tick (optional). `inf-packager` has emitted this for every blueprint
//!   with a BeginPlay handler since P14.5 and nothing here looked it up, so the
//!   handler was compiled, shipped and never called; [`WasmMod::update`] now
//!   runs it through [`WasmMod::init`].
//!
//! The host functions:
//!
//! | import                                            | cap        |
//! |---------------------------------------------------|------------|
//! | `log(ptr: i32, len: i32)`                         | `log`      |
//! | `entity_translation(entity: i64, out: i32) -> i32`| `entities` |
//! | `set_entity_translation(entity: i64, x,y,z: f64)` | `entities` |
//! | `input_is_down(ptr: i32, len: i32) -> i32`        | `input`    |
//! | `spawn_cube(x,y,z: f64) -> i64`                   | `spawn`    |
//!
//! **The out-param ABI (documented choice).** `entity_translation` returns three
//! `f64`s. Rather than depend on wasm multi-value returns, the guest passes a
//! pointer to a 24-byte scratch buffer in *its own* linear memory; the host
//! writes three little-endian `f64`s there and returns `1` (found) or `0`
//! (missing). Strings (`log`, `input_is_down`) are `(ptr, len)` slices the guest
//! owns and the host reads. This keeps every host import a flat scalar signature
//! — the simplest ABI that works, and the one the `inf-mod` shim wraps.
//!
//! # Isolation
//!
//! * **Memory** — a [`wasmtime::StoreLimits`] caps the mod's linear memory; it
//!   cannot grow past [`ExecLimits::memory_bytes`], and it can only ever touch
//!   its own memory (the host reads/writes it explicitly, never the reverse).
//! * **Time** — a mod cannot hang the frame. The default is **fuel**
//!   ([`ExecLimits::fuel_per_update`]), which is *deterministic* (a fixed op
//!   budget, refilled each update) and so honours the §2.5 replay doctrine: an
//!   infinite loop exhausts fuel and traps. A wall-clock **epoch** deadline is
//!   also supported ([`WasmEngine::with_epoch`]) for a real N-ms/frame budget,
//!   but it is non-deterministic and therefore opt-in.
//! * **Faults** — a trapped mod (fuel/epoch/`unreachable`/OOB) is **disabled**
//!   and reported; the host process survives and other mods keep ticking.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use wasmtime::{
    Caller, Config, Engine, Extern, ExternType, Linker, Memory, Module, Store, StoreLimits,
    StoreLimitsBuilder, TypedFunc,
};

use crate::caps::{Capability, ModCaps};
use crate::manifest::ModManifest;
use crate::world::ModWorld;

/// A fat pointer to the embedder's [`ModWorld`], lifetime-erased so it can live
/// in the `'static` store data. It is only ever set for the synchronous duration
/// of a `mod_update` call (see [`WasmMod::update`]) and cleared immediately
/// after, so every host function that dereferences it runs while a real
/// `&mut dyn ModWorld` is live.
type WorldPtr = *mut (dyn ModWorld + 'static);

/// Per-store host state threaded through the linker's closures.
struct ModState {
    /// The resource limiter enforcing the memory cap.
    store_limits: StoreLimits,
    /// The live world for the current call, or `None` outside a call.
    world: Option<WorldPtr>,
}

/// Execution + memory limits applied to every mod in a session.
#[derive(Debug, Clone, Copy)]
pub struct ExecLimits {
    /// Deterministic op budget refilled before each `mod_update`. `None` disables
    /// the fuel limiter (not recommended). An infinite loop exhausts this and
    /// traps.
    pub fuel_per_update: Option<u64>,
    /// Wall-clock epoch deadline (in engine epoch ticks) applied before each
    /// `mod_update`. Only meaningful on an engine built with
    /// [`WasmEngine::with_epoch`] (which advances the epoch on a timer);
    /// `None`/no-ticker → no wall-clock limit.
    pub epoch_deadline: Option<u64>,
    /// Hard cap on a mod's linear memory, in bytes.
    pub memory_bytes: usize,
}

impl Default for ExecLimits {
    fn default() -> Self {
        Self {
            // ~100M ops: generous for a real per-frame mod, trips an infinite loop
            // in well under a frame.
            fuel_per_update: Some(100_000_000),
            epoch_deadline: None,
            // 64 MiB of linear memory.
            memory_bytes: 64 * 1024 * 1024,
        }
    }
}

/// The shared wasmtime engine a mod session compiles + runs against. Owns the
/// optional epoch-advancing timer thread that makes wall-clock deadlines tick.
pub struct WasmEngine {
    engine: Engine,
    /// Whether an epoch ticker is running (deadlines only trip when it is).
    epoch: bool,
    ticker: Option<EpochTicker>,
}

impl WasmEngine {
    /// A deterministic engine: fuel is the only execution limit (the default,
    /// honouring the replay doctrine). No timer thread.
    pub fn new() -> Result<Self, ModError> {
        Ok(Self {
            engine: build_engine(false)?,
            epoch: false,
            ticker: None,
        })
    }

    /// An engine that additionally advances its epoch every `interval`, so
    /// [`ExecLimits::epoch_deadline`] becomes a real wall-clock budget (≈
    /// `deadline * interval`). Non-deterministic — use for interactive editor
    /// sessions, not replay-verified runs.
    pub fn with_epoch(interval: Duration) -> Result<Self, ModError> {
        let engine = build_engine(true)?;
        let ticker = EpochTicker::spawn(engine.clone(), interval);
        Ok(Self {
            engine,
            epoch: true,
            ticker: Some(ticker),
        })
    }

    /// Whether this engine advances its epoch (wall-clock deadlines are live).
    pub fn has_epoch(&self) -> bool {
        self.epoch
    }
}

impl Drop for WasmEngine {
    fn drop(&mut self) {
        // Stop + join the ticker thread deterministically.
        self.ticker.take();
    }
}

fn build_engine(epoch: bool) -> Result<Engine, ModError> {
    let mut cfg = Config::new();
    // Fuel is always available (the deterministic default limiter). Epoch
    // interruption is enabled ONLY on the wall-clock engine: enabling it gives
    // every store a 0 deadline that traps immediately unless a deadline + a
    // ticker are in play, so the fuel-only engine must leave it off.
    cfg.consume_fuel(true);
    if epoch {
        cfg.epoch_interruption(true);
    }
    Engine::new(&cfg).map_err(|e| ModError::Engine(e.to_string()))
}

/// A background thread that bumps the engine epoch on a fixed interval, so epoch
/// deadlines expire in bounded wall-clock time. Stopped + joined on drop.
struct EpochTicker {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl EpochTicker {
    fn spawn(engine: Engine, interval: Duration) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = stop.clone();
        let handle = std::thread::Builder::new()
            .name("inf-wasm-epoch".into())
            .spawn(move || {
                while !stop_thread.load(Ordering::Relaxed) {
                    std::thread::sleep(interval);
                    engine.increment_epoch();
                }
            })
            .expect("spawn epoch ticker");
        Self {
            stop,
            handle: Some(handle),
        }
    }
}

impl Drop for EpochTicker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// One loaded, instantiated mod: its own store + memory + `mod_update` entry.
pub struct WasmMod {
    name: String,
    caps: ModCaps,
    store: Store<ModState>,
    update: TypedFunc<f64, ()>,
    /// The optional one-time `mod_init` entry — a mod's **BeginPlay**.
    ///
    /// `inf-packager` has emitted this export for every blueprint with a
    /// BeginPlay handler since P14.5, and `inf_mod::infinity_mod!` documents it,
    /// but nothing ever looked it up: the handler was compiled, shipped, and
    /// never called. It is `Option` because it is genuinely optional — a mod
    /// with only a Tick handler exports no `mod_init`.
    init: Option<TypedFunc<(), ()>>,
    /// Whether [`init`](Self::init) has already run. A mod's BeginPlay fires
    /// exactly once, immediately before its first tick.
    initialized: bool,
    limits: ExecLimits,
    /// Whether epoch deadlines are live for this mod (engine has a ticker AND a
    /// deadline is configured).
    epoch_active: bool,
    /// Cleared to `false` when the mod traps; a disabled mod is never ticked
    /// again until reloaded.
    enabled: bool,
}

impl WasmMod {
    /// Compile + instantiate `wasm` (binary or, with the `wat` feature, text)
    /// under `caps`. Capability enforcement happens **before** instantiation: if
    /// the module imports a host function whose capability was not granted, this
    /// fails with a capability-anchored [`ModError::MissingCapability`] rather
    /// than a generic link error.
    pub fn instantiate(
        engine: &WasmEngine,
        name: impl Into<String>,
        wasm: &[u8],
        caps: ModCaps,
        limits: ExecLimits,
    ) -> Result<Self, ModError> {
        let name = name.into();
        let module =
            Module::new(&engine.engine, wasm).map_err(|e| ModError::Compile(e.to_string()))?;

        // Deny-by-default: reject any ungranted / unknown import with a clear
        // message before we even build the store.
        check_imports(&module, caps)?;

        let store_limits = StoreLimitsBuilder::new()
            .memory_size(limits.memory_bytes)
            .build();
        let mut store = Store::new(
            &engine.engine,
            ModState {
                store_limits,
                world: None,
            },
        );
        store.limiter(|s| &mut s.store_limits);
        // Seed fuel before instantiation (a fresh consume-fuel store starts at 0,
        // and instantiating a metered module needs a budget).
        if let Some(fuel) = limits.fuel_per_update {
            let _ = store.set_fuel(fuel);
        }

        let linker = build_linker(&engine.engine, caps)?;
        let instance = linker
            .instantiate(&mut store, &module)
            .map_err(|e| ModError::Instantiate(e.to_string()))?;

        let memory = instance.get_memory(&mut store, "memory");
        if memory.is_none() {
            return Err(ModError::MissingExport("memory"));
        }
        let update = instance
            .get_typed_func::<f64, ()>(&mut store, "mod_update")
            .map_err(|_| ModError::MissingExport("mod_update"))?;
        // Optional: a mod with no BeginPlay handler exports no `mod_init`.
        let init = instance
            .get_typed_func::<(), ()>(&mut store, "mod_init")
            .ok();

        let epoch_active = engine.epoch && limits.epoch_deadline.is_some();

        Ok(Self {
            name,
            caps,
            store,
            update,
            init,
            initialized: false,
            limits,
            epoch_active,
            enabled: true,
        })
    }

    /// The mod's display name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Its capability grant.
    pub fn caps(&self) -> ModCaps {
        self.caps
    }

    /// Whether it is still live (a trapped mod is disabled).
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Whether this mod exports a `mod_init` (BeginPlay) entry.
    pub fn has_init(&self) -> bool {
        self.init.is_some()
    }

    /// Whether its one-time `mod_init` has already run.
    pub fn initialized(&self) -> bool {
        self.initialized
    }

    /// Run the mod's one-time `mod_init` (BeginPlay) against `world`, if it has
    /// one and it has not already run.
    ///
    /// Returns `Ok(true)` when the entry actually ran. Called automatically by
    /// [`update`](Self::update) before the first tick, so a host that only ticks
    /// gets BeginPlay for free — one door, per the P22 law, rather than a second
    /// path a caller can forget.
    pub fn init(&mut self, world: &mut dyn ModWorld) -> Result<bool, ModTrap> {
        if !self.enabled {
            return Err(ModTrap::Disabled);
        }
        if self.initialized {
            return Ok(false);
        }
        // **Latched after the decision, not before it** (round-2 LOW). A mod
        // with no `mod_init` export took the early return below with
        // `initialized` already `true`, so the flag said "BeginPlay has run"
        // for a mod whose BeginPlay does not exist — indistinguishable, at
        // every reader, from one that ran. Harmless today because nothing else
        // reads it; the fix is that the flag means what it says.
        let Some(init) = self.init.clone() else {
            return Ok(false);
        };
        self.initialized = true;
        self.refill_budget();
        self.call_with_world(world, |m| init.call(&mut m.store, ()))?;
        Ok(true)
    }

    /// Tick the mod once against `world`, refilling its execution budget first. A
    /// trap (fuel/epoch/`unreachable`/OOB) disables the mod and is returned as
    /// [`ModTrap::Trap`]; the host is unaffected. Ticking a disabled mod is a
    /// no-op returning [`ModTrap::Disabled`].
    ///
    /// Runs [`init`](Self::init) first, once, so a mod's BeginPlay always
    /// precedes its first Tick.
    pub fn update(&mut self, world: &mut dyn ModWorld, dt: f64) -> Result<(), ModTrap> {
        if !self.enabled {
            return Err(ModTrap::Disabled);
        }
        self.init(world)?;
        // Refill the deterministic op budget for this frame. Deliberately AFTER
        // init: BeginPlay gets its own full budget, so a heavy one-time setup
        // cannot starve the first tick.
        self.refill_budget();
        let update = self.update.clone();
        self.call_with_world(world, |m| update.call(&mut m.store, dt))
    }

    /// Refill the per-call deterministic op budget and (re)arm the epoch
    /// deadline.
    fn refill_budget(&mut self) {
        if let Some(fuel) = self.limits.fuel_per_update {
            let _ = self.store.set_fuel(fuel);
        }
        if self.epoch_active {
            if let Some(deadline) = self.limits.epoch_deadline {
                self.store.set_epoch_deadline(deadline);
            }
        }
    }

    /// Publish `world` for the duration of one synchronous guest call, run `f`,
    /// clear the pointer, and turn a trap into a disable.
    ///
    /// SAFETY: `world` is a live `&mut dyn ModWorld` for the whole of `f` (host
    /// imports run only during that call) and the pointer is cleared immediately
    /// after `f` returns — including on the trap path — so no host function ever
    /// dereferences a dangling or aliased world. The transmute only erases the
    /// borrow's lifetime; layout is unchanged.
    fn call_with_world(
        &mut self,
        world: &mut dyn ModWorld,
        f: impl FnOnce(&mut Self) -> wasmtime::Result<()>,
    ) -> Result<(), ModTrap> {
        let raw: *mut dyn ModWorld = world;
        let raw: WorldPtr = unsafe { std::mem::transmute::<*mut dyn ModWorld, WorldPtr>(raw) };
        self.store.data_mut().world = Some(raw);
        let result = f(self);
        self.store.data_mut().world = None;

        match result {
            Ok(()) => Ok(()),
            Err(e) => {
                self.enabled = false;
                Err(ModTrap::Trap(trap_summary(&e)))
            }
        }
    }
}

/// Reduce a wasmtime error to a one-line summary (fuel/epoch/generic).
fn trap_summary(err: &wasmtime::Error) -> String {
    if let Some(trap) = err.downcast_ref::<wasmtime::Trap>() {
        match *trap {
            wasmtime::Trap::OutOfFuel => {
                return "exhausted its fuel budget (hung or too slow)".into()
            }
            wasmtime::Trap::Interrupt => return "exceeded its wall-clock budget (hung)".into(),
            other => return format!("trapped: {other}"),
        }
    }
    err.to_string()
}

/// Scan a module's imports and reject anything not granted by `caps` with a
/// capability-anchored error, before instantiation.
fn check_imports(module: &Module, caps: ModCaps) -> Result<(), ModError> {
    for import in module.imports() {
        // A mod imports only host functions from module "env"; anything else
        // (a memory/table/global import, or a foreign module) is unsupported.
        if import.module() != "env" || !matches!(import.ty(), ExternType::Func(_)) {
            return Err(ModError::UnknownImport(format!(
                "{}::{}",
                import.module(),
                import.name()
            )));
        }
        match ModCaps::cap_for_import(import.name()) {
            None => {
                return Err(ModError::UnknownImport(format!("env::{}", import.name())));
            }
            Some(cap) if !caps.has(cap) => {
                return Err(ModError::MissingCapability {
                    import: import.name().to_owned(),
                    cap,
                });
            }
            Some(_) => {}
        }
    }
    Ok(())
}

/// Build a linker exposing exactly the host functions `caps` grants. Ungranted
/// host functions are simply never defined, so a module that imports one fails
/// to link (and [`check_imports`] has already turned that into a clear error).
fn build_linker(engine: &Engine, caps: ModCaps) -> Result<Linker<ModState>, ModError> {
    let mut linker: Linker<ModState> = Linker::new(engine);
    let wrap = |r: wasmtime::Result<&mut Linker<ModState>>| -> Result<(), ModError> {
        r.map(|_| ()).map_err(|e| ModError::Link(e.to_string()))
    };

    if caps.has(Capability::Log) {
        wrap(linker.func_wrap(
            "env",
            "log",
            |mut caller: Caller<'_, ModState>, ptr: i32, len: i32| {
                let Some(memory) = caller_memory(&mut caller) else {
                    return;
                };
                let (data, state) = memory.data_and_store_mut(&mut caller);
                let Some(world) = state.world else { return };
                let Some(bytes) = slice_of(data, ptr, len) else {
                    return;
                };
                let msg = String::from_utf8_lossy(bytes).into_owned();
                // SAFETY: `world` is live for this call (see WasmMod::update).
                unsafe { (*world).log(&msg) };
            },
        ))?;
    }

    if caps.has(Capability::Entities) {
        wrap(linker.func_wrap(
            "env",
            "entity_translation",
            |mut caller: Caller<'_, ModState>, entity: i64, out_ptr: i32| -> i32 {
                let Some(memory) = caller_memory(&mut caller) else {
                    return 0;
                };
                let (data, state) = memory.data_and_store_mut(&mut caller);
                let Some(world) = state.world else { return 0 };
                // SAFETY: `world` is live for this call.
                let Some(t) = (unsafe { (*world).entity_translation(entity) }) else {
                    return 0;
                };
                let out = out_ptr as usize;
                let Some(end) = out.checked_add(24) else {
                    return 0;
                };
                if end > data.len() {
                    return 0;
                }
                data[out..out + 8].copy_from_slice(&t[0].to_le_bytes());
                data[out + 8..out + 16].copy_from_slice(&t[1].to_le_bytes());
                data[out + 16..out + 24].copy_from_slice(&t[2].to_le_bytes());
                1
            },
        ))?;
        wrap(linker.func_wrap(
            "env",
            "set_entity_translation",
            |caller: Caller<'_, ModState>, entity: i64, x: f64, y: f64, z: f64| {
                if let Some(world) = caller.data().world {
                    // SAFETY: `world` is live for this call.
                    unsafe { (*world).set_entity_translation(entity, [x, y, z]) };
                }
            },
        ))?;
    }

    if caps.has(Capability::Input) {
        wrap(linker.func_wrap(
            "env",
            "input_is_down",
            |mut caller: Caller<'_, ModState>, ptr: i32, len: i32| -> i32 {
                let Some(memory) = caller_memory(&mut caller) else {
                    return 0;
                };
                let (data, state) = memory.data_and_store_mut(&mut caller);
                let Some(world) = state.world else { return 0 };
                let Some(bytes) = slice_of(data, ptr, len) else {
                    return 0;
                };
                let key = String::from_utf8_lossy(bytes).into_owned();
                // SAFETY: `world` is live for this call.
                i32::from(unsafe { (*world).input_is_down(&key) })
            },
        ))?;
    }

    if caps.has(Capability::Spawn) {
        wrap(linker.func_wrap(
            "env",
            "spawn_cube",
            |caller: Caller<'_, ModState>, x: f64, y: f64, z: f64| -> i64 {
                match caller.data().world {
                    // SAFETY: `world` is live for this call.
                    Some(world) => unsafe { (*world).spawn_cube(x, y, z) },
                    None => 0,
                }
            },
        ))?;
    }

    Ok(linker)
}

/// The caller's exported linear memory, if any.
fn caller_memory(caller: &mut Caller<'_, ModState>) -> Option<Memory> {
    match caller.get_export("memory") {
        Some(Extern::Memory(m)) => Some(m),
        _ => None,
    }
}

/// Bounds-checked `data[ptr..ptr+len]`.
fn slice_of(data: &[u8], ptr: i32, len: i32) -> Option<&[u8]> {
    let start = usize::try_from(ptr).ok()?;
    let len = usize::try_from(len).ok()?;
    let end = start.checked_add(len)?;
    data.get(start..end)
}

/// A collection of mods loaded from a directory, ticked together in a
/// deterministic (filename-sorted) order (P14.5, deliverable 3).
pub struct WasmMods {
    engine: WasmEngine,
    limits: ExecLimits,
    mods: Vec<WasmMod>,
    /// Human-readable notes: mods skipped at load, or disabled at runtime.
    reports: Vec<String>,
}

impl WasmMods {
    /// An empty session over a fresh deterministic engine.
    pub fn new(limits: ExecLimits) -> Result<Self, ModError> {
        Ok(Self {
            engine: WasmEngine::new()?,
            limits,
            mods: Vec::new(),
            reports: Vec::new(),
        })
    }

    /// A session over a caller-provided engine (e.g. a wall-clock epoch engine
    /// for the editor).
    pub fn with_engine(engine: WasmEngine, limits: ExecLimits) -> Self {
        Self {
            engine,
            limits,
            mods: Vec::new(),
            reports: Vec::new(),
        }
    }

    /// Load every `.wasm` in `dir` (filename-sorted), each granted the caps in
    /// its sibling `mod.toml` (missing manifest → no caps). A mod that fails to
    /// compile/instantiate is skipped + reported, not fatal. Returns the number
    /// of mods successfully loaded.
    pub fn load_dir(&mut self, dir: &Path) -> Result<usize, ModError> {
        let mut wasm_files: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| ModError::Io(e.to_string()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("wasm"))
            .collect();
        wasm_files.sort();

        let before = self.mods.len();
        for path in wasm_files {
            self.load_one(&path);
        }
        Ok(self.mods.len() - before)
    }

    /// Load a single `.wasm` + its sibling `mod.toml`. Failures are recorded in
    /// [`reports`](Self::reports), never fatal.
    pub fn load_one(&mut self, path: &Path) {
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "mod".into());
        let manifest = match ModManifest::load_beside(path) {
            Ok(m) => m,
            Err(e) => {
                self.reports.push(format!("{name}: bad manifest: {e}"));
                return;
            }
        };
        let display = manifest.name.clone().unwrap_or(name);
        let wasm = match std::fs::read(path) {
            Ok(b) => b,
            Err(e) => {
                self.reports.push(format!("{display}: read failed: {e}"));
                return;
            }
        };
        match WasmMod::instantiate(
            &self.engine,
            display.clone(),
            &wasm,
            manifest.caps,
            self.limits,
        ) {
            Ok(m) => self.mods.push(m),
            Err(e) => self.reports.push(format!("{display}: {e}")),
        }
    }

    /// Discard all loaded mods and reload them from `dir` — the hot-reload path.
    /// Mod state (linear memory) resets on reload; cross-reload state migration
    /// is a documented follow-up (the dylib tier's schema-migration analogue).
    pub fn reload_dir(&mut self, dir: &Path) -> Result<usize, ModError> {
        self.mods.clear();
        self.reports.clear();
        self.load_dir(dir)
    }

    /// Tick every enabled mod once against `world`, in filename order. A mod that
    /// traps is disabled + reported; the rest keep going.
    pub fn tick(&mut self, world: &mut dyn ModWorld, dt: f64) {
        for m in &mut self.mods {
            match m.update(world, dt) {
                Ok(()) | Err(ModTrap::Disabled) => {}
                Err(ModTrap::Trap(msg)) => {
                    let note = format!("mod '{}' disabled: {msg}", m.name());
                    tracing::warn!(target: "inf_wasm_mod", "{note}");
                    self.reports.push(note);
                }
            }
        }
    }

    /// The mods loaded (including any now disabled).
    pub fn mods(&self) -> &[WasmMod] {
        &self.mods
    }

    /// How many mods are still enabled.
    pub fn enabled_count(&self) -> usize {
        self.mods.iter().filter(|m| m.enabled()).count()
    }

    /// Diagnostic notes accumulated (load skips + runtime disables).
    pub fn reports(&self) -> &[String] {
        &self.reports
    }
}

/// A failure loading or instantiating a mod.
#[derive(Debug, thiserror::Error)]
pub enum ModError {
    #[error("building the wasm engine: {0}")]
    Engine(String),
    #[error("compiling the mod: {0}")]
    Compile(String),
    #[error("instantiating the mod: {0}")]
    Instantiate(String),
    #[error("linking a host function: {0}")]
    Link(String),
    #[error(
        "mod requires the `{cap}` capability (imports `env::{import}`) but it was not granted"
    )]
    MissingCapability { import: String, cap: Capability },
    #[error("mod imports an unknown/unsupported symbol `{0}` (not part of the host ABI)")]
    UnknownImport(String),
    #[error("mod is missing the required `{0}` export")]
    MissingExport(&'static str),
    #[error("io error: {0}")]
    Io(String),
}

/// The outcome of ticking a single mod.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ModTrap {
    /// The mod was already disabled (from an earlier trap); it was skipped.
    #[error("mod is disabled")]
    Disabled,
    /// The mod trapped during this tick and has now been disabled.
    #[error("{0}")]
    Trap(String),
}
