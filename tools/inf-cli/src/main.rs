//! Infinity Engine command-line tool.
//!
//! Subcommands (to come): `inf new` (project from template), `inf cook`
//! (build asset packs), `inf bindings` (regenerate ts-rs bindings).

fn main() {
    println!("inf {} (scaffold)", env!("CARGO_PKG_VERSION"));
}
