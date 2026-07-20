//! Plugin-side helpers: implement [`HotComponent`] for your state types and
//! call [`crate::export_plugin!`] once at crate root. Everything here runs
//! *inside* the plugin dylib; the generated `extern "C"` shims catch every
//! panic before it can cross the boundary (an unwinding panic through
//! `extern "C"` aborts the whole process).
//!
//! State crosses reloads as **self-describing JSON**: the new dylib's serde
//! `#[serde(default)]` machinery fills fields the old schema didn't have,
//! which is exactly the "mismatched schema hashes default new fields"
//! migration rule from the roadmap.

use std::marker::PhantomData;
use std::os::raw::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};

use serde::de::DeserializeOwned;
use serde::Serialize;
use xxhash_rust::xxh3::xxh3_64;

use crate::abi::{
    ComponentVTable, LogFn, PluginVTable, WriteFn, ABI_VERSION, LOG_ERROR, LOG_INFO, LOG_WARN,
    STATUS_ERROR, STATUS_OK, STATUS_PANICKED,
};

/// A hot-reloadable component. `Default` supplies fresh-spawn state;
/// serde supplies the reload snapshot format (use `#[serde(default)]` /
/// `#[serde(default = "…")]` on fields added in later versions so old
/// snapshots migrate).
pub trait HotComponent: Serialize + DeserializeOwned + Default + 'static {
    /// Stable component identity across reloads. Reload matches old → new
    /// instances by this name.
    const NAME: &'static str;

    /// A stable, human-maintained schema descriptor (e.g.
    /// `"counter { ticks: i64 }"`). Its xxh3 becomes
    /// [`ComponentVTable::schema_hash`]; change it whenever the state shape
    /// changes so the host can report migrations.
    fn schema() -> &'static str;

    fn tick(&mut self, dt: f64, log: &mut Logger<'_>);
}

/// Borrowed handle to the host's log sink, usable from [`HotComponent::tick`].
pub struct Logger<'a> {
    log: LogFn,
    ctx: *mut c_void,
    _host: PhantomData<&'a ()>,
}

impl Logger<'_> {
    fn emit(&mut self, level: u32, msg: &str) {
        // SAFETY: `log`/`ctx` were handed to us by the host for the duration
        // of this call; the message pointer/len describe a live &str.
        unsafe { (self.log)(self.ctx, level, msg.as_ptr(), msg.len()) }
    }

    pub fn info(&mut self, msg: &str) {
        self.emit(LOG_INFO, msg);
    }

    pub fn warn(&mut self, msg: &str) {
        self.emit(LOG_WARN, msg);
    }

    pub fn error(&mut self, msg: &str) {
        self.emit(LOG_ERROR, msg);
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> &str {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        s
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s
    } else {
        "<non-string panic payload>"
    }
}

unsafe extern "C" fn create_raw<T: HotComponent>() -> *mut c_void {
    match catch_unwind(|| Box::into_raw(Box::<T>::default())) {
        Ok(p) => p.cast(),
        Err(_) => std::ptr::null_mut(),
    }
}

unsafe extern "C" fn destroy_raw<T: HotComponent>(state: *mut c_void) {
    if state.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(state.cast::<T>()))));
}

unsafe extern "C" fn serialize_raw<T: HotComponent>(
    state: *mut c_void,
    write: WriteFn,
    write_ctx: *mut c_void,
) -> u32 {
    let result = catch_unwind(AssertUnwindSafe(|| serde_json::to_vec(&*state.cast::<T>())));
    match result {
        Ok(Ok(bytes)) => {
            write(write_ctx, bytes.as_ptr(), bytes.len());
            STATUS_OK
        }
        Ok(Err(_)) => STATUS_ERROR,
        Err(_) => STATUS_PANICKED,
    }
}

unsafe extern "C" fn deserialize_raw<T: HotComponent>(data: *const u8, len: usize) -> *mut c_void {
    if data.is_null() && len > 0 {
        return std::ptr::null_mut();
    }
    let bytes: &[u8] = if len == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(data, len)
    };
    match catch_unwind(AssertUnwindSafe(|| serde_json::from_slice::<T>(bytes))) {
        Ok(Ok(value)) => Box::into_raw(Box::new(value)).cast(),
        _ => std::ptr::null_mut(),
    }
}

unsafe extern "C" fn tick_raw<T: HotComponent>(
    state: *mut c_void,
    dt: f64,
    log: LogFn,
    log_ctx: *mut c_void,
) -> u32 {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut logger = Logger {
            log,
            ctx: log_ctx,
            _host: PhantomData,
        };
        (*state.cast::<T>()).tick(dt, &mut logger);
    }));
    match result {
        Ok(()) => STATUS_OK,
        Err(payload) => {
            let msg = format!(
                "component '{}' panicked in tick: {}",
                T::NAME,
                panic_message(payload.as_ref())
            );
            log(log_ctx, LOG_ERROR, msg.as_ptr(), msg.len());
            STATUS_PANICKED
        }
    }
}

/// Build the vtable for one component type. Used by [`crate::export_plugin!`].
pub fn component_vtable<T: HotComponent>() -> ComponentVTable {
    ComponentVTable {
        name: T::NAME.as_ptr(),
        name_len: T::NAME.len(),
        schema_hash: xxh3_64(T::schema().as_bytes()),
        create: create_raw::<T>,
        destroy: destroy_raw::<T>,
        serialize: serialize_raw::<T>,
        deserialize: deserialize_raw::<T>,
        tick: tick_raw::<T>,
    }
}

/// Owner of a leaked, immortal [`PluginVTable`]. Stored in a `static
/// OnceLock` by the entry fn — raw pointers inside make the vtable neither
/// `Send` nor `Sync` automatically, but everything they point at is leaked
/// or `'static` string data, so sharing is sound.
pub struct PluginHandle(*const PluginVTable);

// SAFETY: the pointer targets leaked (`'static`) allocations and `'static`
// str data inside this image; the vtable is immutable after construction.
unsafe impl Send for PluginHandle {}
unsafe impl Sync for PluginHandle {}

impl PluginHandle {
    pub fn as_ptr(&self) -> *const PluginVTable {
        self.0
    }
}

/// Assemble and leak the plugin vtable. Called once per process via the
/// entry fn's `OnceLock`; the leak is intentional — the host never unloads
/// plugin images, so their exports are immortal by design.
pub fn build_plugin(version: &'static str, components: Vec<ComponentVTable>) -> PluginHandle {
    let components: &'static [ComponentVTable] = Vec::leak(components);
    let vtable: &'static PluginVTable = Box::leak(Box::new(PluginVTable {
        abi_version: ABI_VERSION,
        version: version.as_ptr(),
        version_len: version.len(),
        component_count: components.len(),
        components: components.as_ptr(),
    }));
    PluginHandle(vtable)
}

/// Export the plugin entry symbol. Call exactly once, at crate root:
///
/// ```ignore
/// inf_hotreload::export_plugin!("1.0.0", [Counter, Fragile]);
/// ```
#[macro_export]
macro_rules! export_plugin {
    ($version:literal, [$($component:ty),* $(,)?]) => {
        #[no_mangle]
        pub extern "C" fn infinity_plugin_entry() -> *const $crate::abi::PluginVTable {
            static PLUGIN: std::sync::OnceLock<$crate::guest::PluginHandle> =
                std::sync::OnceLock::new();
            PLUGIN
                .get_or_init(|| {
                    $crate::guest::build_plugin(
                        $version,
                        vec![$($crate::guest::component_vtable::<$component>()),*],
                    )
                })
                .as_ptr()
        }
    };
}
