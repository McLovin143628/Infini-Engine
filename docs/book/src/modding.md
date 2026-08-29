# Modding

Infini Engine ships a safe, no-compiler extensibility tier for end users: **sandboxed WASM
mods**. Where dylib plugins are ABI-fragile and unsafe, and Blueprints are an authoring tool rather
than a runtime-user surface, WASM mods let a player extend a shipped game safely and without
recompiling the engine. A mod is authored in the *same* Blueprints or Rust as the rest of the game —
"two ways to code, one truth" — and compiled to `.wasm` by the cook target. The `samples/mods`
folder holds a working example.

## What a mod is

A mod is a `.wasm` module plus a `mod.toml` manifest. The sample `spinner/` mod is a few lines of
Rust against the `inf-mod` guest shim: every fixed step it orbits its entity around a circle, and
holding a `boost` input action doubles the rate. It applies to any scene with at least one actor —
it drives the first actor in GUID order, the same entity a Blueprint would see.

## Run a mod

Build the mod to wasm once, drop it (and its manifest) into a mods directory, and point the player
at that directory:

```sh
cargo build --release --target wasm32-unknown-unknown \
    --manifest-path samples/mods/spinner/Cargo.toml
mkdir -p run-mods
cp samples/mods/spinner/target/wasm32-unknown-unknown/release/spinner_mod.wasm run-mods/
cp samples/mods/spinner/mod.toml run-mods/

inf-player --level samples/physics-playground/Playground.inf_lvl --mods run-mods
```

The first actor now orbits, with **no engine recompile**. Delete the `.wasm` and it stops; edit and
rebuild it and the editor hot-reloads it live in Simulate.

## Cook a mod from a Blueprint

You do not have to hand-write Rust. Author a mod as a Blueprint class and let the cook target
transpile and build it through the *same* codegen the compiled game path uses:

```sh
inf cook --mods MyMod.inf_act --out run-mods
```

This lifts the class through the transpiler, wraps it in the mod template with the host shim, and
builds it to `wasm32-unknown-unknown`.

## The capability model

Mods are **deny-by-default**. A `mod.toml` beside the `.wasm` lists exactly the capabilities the
sandbox will link; a missing manifest grants nothing, and a mod that imports a host function it was
not granted **fails to load** with a clear message. The capabilities are narrow and explicit:

| capability | host functions unlocked            |
|------------|------------------------------------|
| `entities` | read / write entity transforms     |
| `input`    | query held input actions           |
| `log`      | emit log lines                     |
| `spawn`    | spawn new entities                 |

For the full security posture, see the modding memo
([`docs/memos/p14-wasm-modding.md`](https://github.com/McLovin143628/Infini-Engine/blob/main/docs/memos/p14-wasm-modding.md)).
