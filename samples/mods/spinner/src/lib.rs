//! The `spinner` sample mod (ROADMAP P14.5, deliverable 2/4).
//!
//! A few lines of Rust against the `inf-mod` shim: every fixed step it advances
//! an angle and drives entity 1 around a circle, so a scene it is applied to has
//! its actor orbit with no engine recompile. Holding the `"boost"` input action
//! doubles the rate — proving the `input` capability too.
//!
//! Built for `wasm32-unknown-unknown` (`crate-type = ["cdylib"]`); its
//! capabilities are granted by the sibling `mod.toml` (`entities`, `input`,
//! `log`). This is exactly the shape the WASM cook target emits when it lowers a
//! Blueprint through the transpiler — hand-written here so the end-to-end path is
//! provable without the full node-kit lowering.

use std::cell::Cell;

thread_local! {
    /// The current orbit angle (radians). Single-threaded wasm ⇒ a `Cell` is all
    /// the persistence a mod needs across ticks.
    static ANGLE: Cell<f64> = const { Cell::new(0.0) };
}

/// The entity this mod drives (the sim assigns actors stable ids in `Guid`
/// order; 1 is the first actor — the same id a blueprint would see).
const ENTITY: i64 = 1;
/// Orbit radius (world units).
const RADIUS: f64 = 2.0;
/// Base angular speed (radians/second).
const SPEED: f64 = 1.5;

fn update(dt: f64) {
    let rate = if inf_mod::input_is_down("boost") {
        SPEED * 2.0
    } else {
        SPEED
    };
    let angle = ANGLE.with(|a| {
        let next = a.get() + rate * dt;
        a.set(next);
        next
    });
    inf_mod::set_translation(ENTITY, [RADIUS * angle.cos(), 1.0, RADIUS * angle.sin()]);
}

fn init() {
    inf_mod::log("spinner mod loaded");
}

inf_mod::infinity_mod!(update = update, init = init);
