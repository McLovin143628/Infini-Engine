//! Infinity Engine standalone player.
//!
//! Loads a cooked asset pack and runs the game loop. Also serves as the
//! play-in-editor (PIE) subprocess. Headless mode (`--run-frames N`) is used
//! by CI smoke tests.

fn main() {
    println!("inf-player {} (scaffold)", env!("CARGO_PKG_VERSION"));
}
