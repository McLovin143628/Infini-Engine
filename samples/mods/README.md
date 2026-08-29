# Sample mods — the moddable game (P14.5)

Infini Engine loads sandboxed **WASM mods** at runtime: safe, no-compiler,
end-user extensibility that neither dylib plugins (ABI-fragile / unsafe) nor
Blueprints (not runtime-user-facing) provide. A mod is authored in the **same**
Blueprints/Rust as the rest of the game — no new scripting language, "two ways to
code, one truth" — and compiled blueprint → Rust → `.wasm` by the cook target.

## What's here

- **`spinner/`** — the sample mod (a few lines of Rust against the `inf-mod`
  guest shim). Every fixed step it orbits its entity around a circle; holding the
  `boost` input action doubles the rate. Its `mod.toml` grants exactly the
  capabilities it uses (`entities`, `input`, `log`).

## Run it

The spinner applies to **any scene with at least one actor** — it drives entity
id `1` (the first actor in `Guid` order, the same id a Blueprint sees). Point the
player at a mods directory:

```sh
# 1. Build the mod to wasm (once):
cargo build --release --target wasm32-unknown-unknown \
    --manifest-path samples/mods/spinner/Cargo.toml
mkdir -p run-mods
cp samples/mods/spinner/target/wasm32-unknown-unknown/release/spinner_mod.wasm run-mods/
cp samples/mods/spinner/mod.toml run-mods/

# 2. Play a scene with the mod loaded:
inf-player --level samples/physics-playground/Main.inf_lvl --mods run-mods
```

The first actor now orbits — with **no engine recompile**. Delete the `.wasm`
and it stops; edit + rebuild it and the editor hot-reloads it live in Simulate.

## Cook a mod from a Blueprint

Instead of hand-writing Rust, author a mod as a Blueprint class and let the cook
target transpile + build it:

```sh
inf cook --mods MyMod.inf_act --out run-mods
```

This transpiles the class through the **same** codegen the compiled path uses,
wraps it in the `cdylib` mod template with the host shim, and builds it to
`wasm32-unknown-unknown` (or prints install instructions if the target is
missing). See `runtime/inf-packager/src/mods.rs`.

## The capability model

Mods are **deny-by-default**. A `mod.toml` beside the `.wasm` lists the
capabilities the sandbox will link; a missing manifest grants nothing. A mod that
imports a host function it wasn't granted **fails to load** with a clear message.
See `docs/memos/p14-wasm-modding.md` for the full security posture.

| capability | host functions unlocked                        |
|------------|------------------------------------------------|
| `entities` | read / write entity transforms                 |
| `input`    | query held input actions                       |
| `log`      | emit log lines                                 |
| `spawn`    | spawn new entities                             |
