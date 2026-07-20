//! Host side: load plugin dylibs, own component instances, tick them with
//! panic containment, and carry state across reloads.
//!
//! Two standing rules (ROADMAP §5, Spike C):
//!
//! - **Never unload.** Unloading a Rust dylib is UB-adjacent (TLS
//!   destructors, leaked `'static` references, vtables still in use), so
//!   every loaded [`libloading::Library`] is leaked. Old code stays mapped;
//!   reload just swaps which vtable new work goes through.
//! - **Never load the build artifact in place.** Windows locks a loaded DLL
//!   on disk, which would break the *next* `cargo build`. The host copies
//!   the file to a content-addressed shadow path
//!   (`{stem}.{xxh3 of bytes}.{ext}`) and loads that; the original stays
//!   writable, and the hash makes re-loading an unchanged file a no-op.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::os::raw::c_void;
use std::path::{Path, PathBuf};

use xxhash_rust::xxh3::xxh3_64;

use crate::abi::{
    ComponentVTable, EntryFn, PluginVTable, ABI_VERSION, ENTRY_SYMBOL, LOG_ERROR, STATUS_OK,
    STATUS_PANICKED,
};

#[derive(Debug, thiserror::Error)]
pub enum HostError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to load dylib: {0}")]
    Load(#[from] libloading::Error),
    #[error("plugin entry returned a null vtable")]
    NullVTable,
    #[error("plugin ABI version {found} does not match host ABI version {ABI_VERSION}")]
    AbiMismatch { found: u32 },
    #[error("plugin exports a non-UTF-8 string")]
    BadString,
    #[error("plugin exports duplicate component '{0}'")]
    DuplicateComponent(String),
    #[error("plugin has no component named '{0}'")]
    UnknownComponent(String),
    #[error("component '{0}' failed to create default state")]
    CreateFailed(String),
    #[error("component '{0}' failed to serialize (status {1})")]
    SerializeFailed(String, u32),
    #[error("no instance with id {0:?}")]
    UnknownInstance(InstanceId),
    #[error("instance state is not valid JSON: {0}")]
    BadState(#[from] serde_json::Error),
}

/// A validated, immortal plugin image.
pub struct Plugin {
    version: String,
    content_hash: u64,
    shadow_path: Option<PathBuf>,
    components: BTreeMap<String, &'static ComponentVTable>,
}

impl Plugin {
    /// Validate a raw vtable into a [`Plugin`].
    ///
    /// # Safety
    /// `vtable` (and everything it points to — version string, component
    /// array, component names, fn pointers) must be valid for the rest of
    /// the program. The loader guarantees this by leaking the `Library`;
    /// in-process tests guarantee it by leaking the vtable.
    pub unsafe fn from_vtable(
        vtable: &'static PluginVTable,
        content_hash: u64,
        shadow_path: Option<PathBuf>,
    ) -> Result<Self, HostError> {
        if vtable.abi_version != ABI_VERSION {
            return Err(HostError::AbiMismatch {
                found: vtable.abi_version,
            });
        }
        let version = str_from_raw(vtable.version, vtable.version_len)?.to_owned();
        let raw_components = if vtable.component_count == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(vtable.components, vtable.component_count)
        };
        let mut components = BTreeMap::new();
        for component in raw_components {
            let name = str_from_raw(component.name, component.name_len)?.to_owned();
            if components.insert(name.clone(), component).is_some() {
                return Err(HostError::DuplicateComponent(name));
            }
        }
        Ok(Plugin {
            version,
            content_hash,
            shadow_path,
            components,
        })
    }

    pub fn version(&self) -> &str {
        &self.version
    }

    /// xxh3 of the dylib bytes (0 for in-process test plugins).
    pub fn content_hash(&self) -> u64 {
        self.content_hash
    }

    /// Where the loaded image actually lives (the shadow copy).
    pub fn shadow_path(&self) -> Option<&Path> {
        self.shadow_path.as_deref()
    }

    pub fn component(&self, name: &str) -> Option<&'static ComponentVTable> {
        self.components.get(name).copied()
    }

    pub fn component_names(&self) -> impl Iterator<Item = &str> {
        self.components.keys().map(String::as_str)
    }
}

unsafe fn str_from_raw(ptr: *const u8, len: usize) -> Result<&'static str, HostError> {
    if ptr.is_null() && len > 0 {
        return Err(HostError::BadString);
    }
    let bytes: &'static [u8] = if len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ptr, len)
    };
    std::str::from_utf8(bytes).map_err(|_| HostError::BadString)
}

/// Loads plugin dylibs through content-addressed shadow copies.
pub struct PluginHost {
    shadow_dir: PathBuf,
}

impl PluginHost {
    pub fn new(shadow_dir: impl Into<PathBuf>) -> Result<Self, HostError> {
        let shadow_dir = shadow_dir.into();
        fs::create_dir_all(&shadow_dir)?;
        Ok(PluginHost { shadow_dir })
    }

    /// Copy `path` to its shadow location, load it, and validate the entry.
    /// Loading the same file contents twice reuses the same shadow copy.
    pub fn load(&self, path: &Path) -> Result<Plugin, HostError> {
        let bytes = fs::read(path)?;
        let hash = xxh3_64(&bytes);
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("plugin");
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("bin");
        let shadow = self.shadow_dir.join(format!("{stem}.{hash:016x}.{ext}"));
        if !shadow.exists() {
            // Write via temp + rename so a concurrent load of the same
            // contents can't observe a half-written shadow.
            let tmp = self
                .shadow_dir
                .join(format!("{stem}.{hash:016x}.{}.tmp", std::process::id()));
            fs::write(&tmp, &bytes)?;
            if let Err(err) = fs::rename(&tmp, &shadow) {
                let _ = fs::remove_file(&tmp);
                if !shadow.exists() {
                    return Err(err.into());
                }
            }
        }
        // Immortal by doctrine: never unload (see module docs).
        let library: &'static libloading::Library =
            Box::leak(Box::new(unsafe { libloading::Library::new(&shadow) }?));
        let entry: libloading::Symbol<'static, EntryFn> = unsafe { library.get(ENTRY_SYMBOL) }?;
        let vtable_ptr = unsafe { entry() };
        if vtable_ptr.is_null() {
            return Err(HostError::NullVTable);
        }
        // SAFETY: the image is leaked, so its exports live forever.
        unsafe { Plugin::from_vtable(&*vtable_ptr, hash, Some(shadow)) }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstanceId(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstanceStatus {
    Running,
    /// Ticking is suspended after a panic/error until the next successful
    /// reload of the component.
    Disabled,
}

#[derive(Debug, Clone)]
pub struct LogRecord {
    /// One of the `abi::LOG_*` levels.
    pub level: u32,
    pub message: String,
}

/// Outcome of [`HotWorld::reload`], per component instance.
#[derive(Debug, Default)]
pub struct ReloadReport {
    /// Same schema hash: state carried over verbatim.
    pub carried: Vec<InstanceId>,
    /// Schema hash changed: state migrated through the self-describing
    /// format (new fields defaulted).
    pub migrated: Vec<InstanceId>,
    /// The new plugin no longer exports this component; the instance keeps
    /// running the old (still-loaded) code.
    pub removed: Vec<InstanceId>,
    /// Serialize/deserialize failed; the instance keeps its old code+state.
    pub failed: Vec<(InstanceId, String)>,
}

struct Instance {
    name: String,
    vtable: &'static ComponentVTable,
    state: *mut c_void,
    status: InstanceStatus,
    last_error: Option<String>,
}

/// Owns live component instances. Single-threaded by design (the editor's
/// script tick runs on one thread); holds raw state pointers, so it is
/// deliberately `!Send`.
#[derive(Default)]
pub struct HotWorld {
    instances: Vec<Instance>,
    logs: Vec<LogRecord>,
}

impl fmt::Debug for HotWorld {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HotWorld")
            .field("instances", &self.instances.len())
            .field("logs", &self.logs.len())
            .finish()
    }
}

unsafe extern "C" fn host_write(ctx: *mut c_void, data: *const u8, len: usize) {
    if len == 0 {
        return;
    }
    let buffer = &mut *ctx.cast::<Vec<u8>>();
    buffer.extend_from_slice(std::slice::from_raw_parts(data, len));
}

unsafe extern "C" fn host_log(ctx: *mut c_void, level: u32, msg: *const u8, len: usize) {
    let logs = &mut *ctx.cast::<Vec<LogRecord>>();
    let bytes = if len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(msg, len)
    };
    logs.push(LogRecord {
        level,
        message: String::from_utf8_lossy(bytes).into_owned(),
    });
}

impl HotWorld {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a default-state instance of `component` from `plugin`.
    pub fn spawn(&mut self, plugin: &Plugin, component: &str) -> Result<InstanceId, HostError> {
        let vtable = plugin
            .component(component)
            .ok_or_else(|| HostError::UnknownComponent(component.to_owned()))?;
        // SAFETY: vtable comes from a validated, immortal Plugin.
        let state = unsafe { (vtable.create)() };
        if state.is_null() {
            return Err(HostError::CreateFailed(component.to_owned()));
        }
        self.instances.push(Instance {
            name: component.to_owned(),
            vtable,
            state,
            status: InstanceStatus::Running,
            last_error: None,
        });
        Ok(InstanceId(self.instances.len() - 1))
    }

    /// Tick every running instance. A panicking/erroring component is
    /// disabled (not ticked again until a reload fixes it) and its message
    /// lands in [`Self::logs`]; everything else keeps running.
    pub fn tick(&mut self, dt: f64) {
        let logs = &mut self.logs;
        for instance in &mut self.instances {
            if instance.status != InstanceStatus::Running {
                continue;
            }
            // SAFETY: state was produced by this same vtable; log ctx is a
            // live &mut Vec<LogRecord> for the duration of the call.
            let status = unsafe {
                (instance.vtable.tick)(
                    instance.state,
                    dt,
                    host_log,
                    (logs as *mut Vec<LogRecord>).cast(),
                )
            };
            if status != STATUS_OK {
                instance.status = InstanceStatus::Disabled;
                let detail = logs
                    .iter()
                    .rev()
                    .find(|r| r.level == LOG_ERROR)
                    .map(|r| r.message.clone())
                    .unwrap_or_else(|| format!("tick returned status {status}"));
                instance.last_error = Some(if status == STATUS_PANICKED {
                    detail
                } else {
                    format!("tick failed (status {status}): {detail}")
                });
            }
        }
    }

    /// Serialize one instance's state (the same bytes a reload would carry).
    pub fn serialize_instance(&self, id: InstanceId) -> Result<Vec<u8>, HostError> {
        let instance = self
            .instances
            .get(id.0)
            .ok_or(HostError::UnknownInstance(id))?;
        let mut buffer = Vec::new();
        // SAFETY: state/vtable pairing upheld; write ctx is a live Vec.
        let status = unsafe {
            (instance.vtable.serialize)(
                instance.state,
                host_write,
                (&mut buffer as *mut Vec<u8>).cast(),
            )
        };
        if status != STATUS_OK {
            return Err(HostError::SerializeFailed(instance.name.clone(), status));
        }
        Ok(buffer)
    }

    /// The instance state as JSON — the editor's inspector view of script
    /// state, and what the tests assert against.
    pub fn state_json(&self, id: InstanceId) -> Result<serde_json::Value, HostError> {
        Ok(serde_json::from_slice(&self.serialize_instance(id)?)?)
    }

    /// Swap every instance over to `new_plugin`: old code serializes, new
    /// code deserializes (defaulting fields the old schema lacked), old
    /// state is destroyed by the old dylib (still loaded). Instances the new
    /// plugin doesn't export, or whose state transfer fails, keep running
    /// the old code. A successful swap re-enables a disabled instance —
    /// reloading fixed code is the whole point.
    pub fn reload(&mut self, new_plugin: &Plugin) -> ReloadReport {
        let mut report = ReloadReport::default();
        for (index, instance) in self.instances.iter_mut().enumerate() {
            let id = InstanceId(index);
            let Some(new_vtable) = new_plugin.component(&instance.name) else {
                report.removed.push(id);
                continue;
            };
            let mut buffer = Vec::new();
            // SAFETY: old state/vtable pairing upheld.
            let status = unsafe {
                (instance.vtable.serialize)(
                    instance.state,
                    host_write,
                    (&mut buffer as *mut Vec<u8>).cast(),
                )
            };
            if status != STATUS_OK {
                report
                    .failed
                    .push((id, format!("serialize failed with status {status}")));
                continue;
            }
            // SAFETY: new vtable comes from a validated, immortal Plugin.
            let new_state = unsafe { (new_vtable.deserialize)(buffer.as_ptr(), buffer.len()) };
            if new_state.is_null() {
                report
                    .failed
                    .push((id, "deserialize returned null".to_owned()));
                continue;
            }
            let migrated = new_vtable.schema_hash != instance.vtable.schema_hash;
            // SAFETY: destroy old state with the vtable that created it; the
            // old dylib is never unloaded, so the call stays valid.
            unsafe { (instance.vtable.destroy)(instance.state) };
            instance.vtable = new_vtable;
            instance.state = new_state;
            instance.status = InstanceStatus::Running;
            instance.last_error = None;
            if migrated {
                report.migrated.push(id);
            } else {
                report.carried.push(id);
            }
        }
        report
    }

    pub fn status(&self, id: InstanceId) -> Option<InstanceStatus> {
        self.instances.get(id.0).map(|i| i.status)
    }

    pub fn last_error(&self, id: InstanceId) -> Option<&str> {
        self.instances.get(id.0)?.last_error.as_deref()
    }

    pub fn logs(&self) -> &[LogRecord] {
        &self.logs
    }

    pub fn take_logs(&mut self) -> Vec<LogRecord> {
        std::mem::take(&mut self.logs)
    }
}

impl Drop for HotWorld {
    fn drop(&mut self) {
        for instance in &self.instances {
            // SAFETY: destroy with the creating vtable; images are immortal.
            unsafe { (instance.vtable.destroy)(instance.state) };
        }
    }
}
