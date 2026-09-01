# Web player (WebGPU / wasm32) — P14.2

The standalone player (`runtime/inf-player`) cross-compiles to
`wasm32-unknown-unknown` and runs in a WebGPU browser. The browser build reuses
the **entire** runtime — the same fixed-step loop, `inf-render` forward renderer,
`inf-scene`/`inf-asset` pack loader, and blueprint interpreter as desktop — behind
`#[cfg(target_arch = "wasm32")]` seams. There is **no separate web engine**: a
game runs in Chrome exactly as it does on the desktop player.

## What CI verifies

The `wasm-check` CI job runs, on every push:

```sh
rustup target add wasm32-unknown-unknown
RUSTFLAGS='--cfg getrandom_backend="wasm_js"' \
  cargo check --target wasm32-unknown-unknown -p inf-player
```

This is the real, enforced gate: the player's whole dependency tree must compile
for the browser's target. (It is a `check`, not a live run — CI has no headless
WebGPU device, exactly as the desktop GPU paths are compile-checked but
human-verified.)

## How the wasm build is made portable

The player's desktop dep tree contains native/threaded crates that do not target
`wasm32-unknown-unknown`. Each is handled at the crate that owns it, so **desktop
behaviour is byte-for-byte unchanged**:

| Dependency | Desktop | wasm32 |
|---|---|---|
| `uuid` v4 randomness | OS RNG | `js` feature (browser `crypto`) |
| `getrandom` | OS RNG | `wasm_js` backend (`--cfg getrandom_backend="wasm_js"`) |
| `zstd` (pack codec, `inf-asset`) | C `zstd` (encode + decode) | pure-Rust **`ruzstd`** (decode only) |
| `notify` file watcher (`inf-asset`) | on | **off** (no browser file watcher) |
| `meshopt` meshlet builder (`inf-vgeom`) | on (cook-time) | **off** (browser only *loads* cooked DAGs) |
| `wgpu` backend | Vulkan/Metal | `webgpu` |
| `winit` surface | native window | `<canvas>` via `WindowAttributesExtWebSys` |
| `std::time::Instant` | std | `web-time` (`performance.now()`) |

Cook-only code paths (pack *writing*, meshlet *building*, file watching) are
gated out of wasm — the browser never cooks; it fetches a pack cooked on desktop.

## The entry point

`inf_player::web::start_player(canvas_id, pack_url)` is exported to JS via
wasm-bindgen:

```js
import init, { start_player } from "./inf_player.js";
await init();
start_player("game", "./content.ipack");
```

It fetches the pack over HTTP (whole-file `fetch` v1), parses it with
`PackReader::from_bytes` (ruzstd-decoding blobs), builds the world through the
**same** `build_world_from_pack` path the desktop `--pack` boot uses, and runs the
winit web event loop on a WebGPU surface bound to the `<canvas>`.

## Building + running it yourself (human-verified)

CI checks compilation; a real run is a human step:

```sh
# 1. tools (once)
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli

# 2. cook a project + emit the web bundle skeleton (index.html + pack + instructions),
#    which also runs the two steps below when the tools are on PATH:
cargo run -p inf-cli -- export --project <your-project> --target web --out ./web

# …or run the two steps manually:
RUSTFLAGS='--cfg getrandom_backend="wasm_js"' \
  cargo build --release --target wasm32-unknown-unknown -p inf-player
wasm-bindgen --target web --no-typescript --out-name inf_player \
  --out-dir ./web target/wasm32-unknown-unknown/release/inf_player.wasm

# 3. serve over HTTP (WebGPU + module scripts require a real origin, not file://)
cd web && python -m http.server 8080   # open http://localhost:8080 in Chrome 113+
```

## Honest runtime status (the remaining seam)

The wasm player **compiles and loads**; the one runtime step still to land before
a live in-browser frame is **async GPU-adapter acquisition**. `inf-render`'s
`GpuContext::for_surface` requests its adapter/device with `pollster::block_on`,
which is correct on desktop but **cannot block the browser's single main thread**.
A live WebGPU run needs an async adapter path:

```rust
// today (desktop-correct, wasm-blocking):
let adapter = pollster::block_on(instance.request_adapter(&opts))?;
// needed for the browser: await request_adapter / request_device in start_player,
// then hand the ready GpuContext to the winit web loop.
```

This is a bounded, tracked follow-up in `inf-render` + `web::start_player`
(the fetch, pack decode, world build, canvas surface, and fixed-step loop are all
already wired). We do **not** claim a working in-browser render until that async
seam lands and a human has verified a frame in Chrome.

### Other documented follow-ups

- **Pack streaming**: v1 fetches the whole pack; HTTP range-request streaming
  (index first, blobs on demand) is a follow-up.
- **Asset-mesh geometry**: like PIE, web v1 renders `MeshRef.asset` entities as
  placeholder cubes (it streams no `.inf_vmesh`); wiring the pack's vmeshes into
  the web render host mirrors the desktop follow-up.
- **Touch layout**: the default on-screen controls (a left virtual stick + a jump
  button) use a fixed 1280×720 reference layout; resolution/safe-area awareness is
  a follow-up (see `crates/inf-input/src/touch.rs`).
