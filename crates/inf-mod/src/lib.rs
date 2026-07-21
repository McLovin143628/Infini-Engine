//! Guest-side shim for Infinity Engine WASM mods (ROADMAP P14.5, deliverable 2).
//!
//! A mod is a `cdylib` compiled to `wasm32-unknown-unknown`. It imports the host
//! functions its capabilities grant and exports `mod_update(dt)`. This crate
//! wraps that raw `extern "C"` ABI (see [`inf-wasm-host`]) in safe Rust and
//! provides the [`infinity_mod!`] entry macro, so a mod is a few lines:
//!
//! ```ignore
//! use std::cell::Cell;
//! thread_local! { static ANGLE: Cell<f64> = const { Cell::new(0.0) }; }
//!
//! fn update(dt: f64) {
//!     let a = ANGLE.with(|c| { let v = c.get() + dt; c.set(v); v });
//!     inf_mod::set_translation(1, [a.cos() * 2.0, 1.0, a.sin() * 2.0]);
//! }
//! inf_mod::infinity_mod!(update = update);
//! ```
//!
//! # ABI
//!
//! On `wasm32` the wrappers call real host imports (module `"env"`). On any other
//! target — so the workspace's `cargo check`/clippy/fmt cover this crate — they
//! compile to inert stubs (a mod is never *run* natively). The out-param for
//! [`translation`] uses a 24-byte stack buffer, matching the host's documented
//! little-endian `f64` ×3 layout.
//!
//! [`inf-wasm-host`]: ../inf_wasm_host/index.html

#![allow(clippy::missing_safety_doc)]

// ── The raw host imports (wasm32 only) ───────────────────────────────────────
#[cfg(target_arch = "wasm32")]
mod ffi {
    #[link(wasm_import_module = "env")]
    extern "C" {
        pub fn log(ptr: *const u8, len: usize);
        pub fn entity_translation(entity: i64, out_ptr: *mut u8) -> i32;
        pub fn set_entity_translation(entity: i64, x: f64, y: f64, z: f64);
        pub fn input_is_down(ptr: *const u8, len: usize) -> i32;
        pub fn spawn_cube(x: f64, y: f64, z: f64) -> i64;
    }
}

/// Emit a log line to the host (capability `log`).
pub fn log(message: &str) {
    #[cfg(target_arch = "wasm32")]
    // SAFETY: the pointer/len name a live UTF-8 slice the host only reads.
    unsafe {
        ffi::log(message.as_ptr(), message.len());
    }
    #[cfg(not(target_arch = "wasm32"))]
    let _ = message;
}

/// Read `entity`'s world-space translation, or `None` if it does not exist
/// (capability `entities`).
pub fn translation(entity: i64) -> Option<[f64; 3]> {
    #[cfg(target_arch = "wasm32")]
    // SAFETY: `buf` is a live 24-byte region the host writes exactly 3 LE f64s
    // into when it returns 1.
    unsafe {
        let mut buf = [0u8; 24];
        if ffi::entity_translation(entity, buf.as_mut_ptr()) == 0 {
            return None;
        }
        let f = |i: usize| f64::from_le_bytes(buf[i..i + 8].try_into().unwrap());
        Some([f(0), f(8), f(16)])
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = entity;
        None
    }
}

/// Set `entity`'s world-space translation (capability `entities`).
pub fn set_translation(entity: i64, t: [f64; 3]) {
    #[cfg(target_arch = "wasm32")]
    // SAFETY: a plain scalar host call.
    unsafe {
        ffi::set_entity_translation(entity, t[0], t[1], t[2]);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (entity, t);
    }
}

/// Whether input action `key` is currently held (capability `input`).
pub fn input_is_down(key: &str) -> bool {
    #[cfg(target_arch = "wasm32")]
    // SAFETY: the pointer/len name a live UTF-8 slice the host only reads.
    unsafe {
        ffi::input_is_down(key.as_ptr(), key.len()) != 0
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = key;
        false
    }
}

/// Spawn a unit cube at `(x, y, z)`, returning its new entity id (capability
/// `spawn`).
pub fn spawn_cube(x: f64, y: f64, z: f64) -> i64 {
    #[cfg(target_arch = "wasm32")]
    // SAFETY: a plain scalar host call.
    unsafe {
        ffi::spawn_cube(x, y, z)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (x, y, z);
        0
    }
}

/// Define a mod's exported entry points. Wraps an `fn update(dt: f64)` (and,
/// optionally, `fn init()`) in the `#[no_mangle] extern "C"` exports the host
/// looks up (`mod_update`, `mod_init`).
///
/// ```ignore
/// fn update(dt: f64) { /* ... */ }
/// inf_mod::infinity_mod!(update = update);
/// ```
#[macro_export]
macro_rules! infinity_mod {
    (update = $update:path $(, init = $init:path)? $(,)?) => {
        /// The per-fixed-step entry the host calls.
        #[no_mangle]
        pub extern "C" fn mod_update(dt: f64) {
            let update: fn(f64) = $update;
            update(dt);
        }
        $(
            /// The one-time init entry (optional).
            #[no_mangle]
            pub extern "C" fn mod_init() {
                let init: fn() = $init;
                init();
            }
        )?
    };
}
