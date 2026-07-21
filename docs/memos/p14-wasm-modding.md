# Memo — P14.5 sandboxed WASM modding: security posture, trust model & cook flow

**Status:** BUILT (Phase 14.5, closing Phase 14). This memo records the sandbox's
guarantees, its non-guarantees, the trust model, and the author→ship flow so they
are not relitigated. It complements the crossref memo
(`rust-report-crossref.md`, Amendment 2), which decided *why* WASM (safe,
no-compiler, end-user extensibility) instead of a fourth scripting language.

## The pieces

| crate / path                         | role                                                              |
|--------------------------------------|-------------------------------------------------------------------|
| `crates/inf-wasm-host`               | the sandbox: `wasmtime` engine, capability-scoped linker, loader  |
| `crates/inf-mod`                     | the guest shim mods author against (safe wrappers + entry macro)  |
| `runtime/inf-packager/src/mods.rs`   | the WASM cook target (Blueprint → Rust → `cdylib` → `.wasm`)       |
| `runtime/inf-player/src/mods.rs`     | the player's mod loader + `EcsWorld` adapter (`--mods <dir>`)      |
| `editor/crates/inf-editor-core/src/mods.rs` | the editor `ModsSession` + notify hot-reload               |
| `samples/mods/spinner`               | the sample mod + the moddable-game story                          |

## The host ABI (v1)

A mod is a plain **core** wasm module (no component model, no WASI) behind a flat
`i64`/`f64`/`i32` ABI. It **exports** `memory` and `mod_update(dt: f64)` (and
optionally `mod_init()`); it **imports** (module `"env"`) only the host functions
its capabilities unlock:

| import                                             | capability |
|----------------------------------------------------|------------|
| `log(ptr, len)`                                    | `log`      |
| `entity_translation(entity, out_ptr) -> i32`       | `entities` |
| `set_entity_translation(entity, x, y, z)`          | `entities` |
| `input_is_down(ptr, len) -> i32`                   | `input`    |
| `spawn_cube(x, y, z) -> i64`                        | `spawn`    |

**Out-param ABI (documented choice).** `entity_translation` returns three `f64`s
via a guest-supplied 24-byte scratch pointer (three little-endian `f64`s written
by the host, `1`/`0` return for found/missing) rather than wasm multi-value — the
simplest ABI that keeps every import a flat scalar signature. Strings are
`(ptr, len)` slices the guest owns and the host reads. The embedder implements one
trait — `ModWorld` — the exact analogue of the Blueprint interpreter's `Host`.

## What the sandbox **guarantees**

1. **Memory isolation.** A mod can only touch its own linear memory. The host
   reads/writes that memory explicitly (never the reverse), and a
   `wasmtime::StoreLimits` caps its size (`ExecLimits::memory_bytes`, default
   64 MiB) — it cannot grow past the cap or reach host/other-mod memory.
2. **Capability scoping, deny-by-default.** Only the host functions a mod's
   `ModCaps` grant are linked. Grants come from a `mod.toml` beside the `.wasm`;
   a **missing manifest grants nothing**. A module importing an ungranted (or
   unknown) host function **fails to instantiate** with a capability-anchored
   message — enforced by scanning imports *before* instantiation, so the failure
   is a clear "requires `entities`" rather than a generic link error.
3. **Bounded execution — a mod cannot hang the frame.** The default limiter is
   **fuel** (`ExecLimits::fuel_per_update`, refilled each update): a fixed,
   **deterministic** op budget, so it honours the §2.5 replay doctrine. An
   infinite loop exhausts fuel and traps. A wall-clock **epoch** deadline
   (`WasmEngine::with_epoch`, an N-ms/frame budget via a timer thread) is also
   supported for interactive editor sessions but is non-deterministic and so
   opt-in. Both are tested against a deliberately infinite-looping module
   (`inf-wasm-host` unit tests `hung_mod_traps_on_fuel…` / `…on_wall_clock_epoch`).
4. **Fault containment.** A trapped mod (fuel/epoch/`unreachable`/out-of-bounds)
   is **disabled and reported**; the host process survives and every other mod
   keeps ticking. Verified: after a hang traps, a good mod on the same engine
   still runs.
5. **Determinism (fuel path).** Fuel + a deterministic `spawn_cube` Guid (derived
   from the entity id, never `v4`) keep a modded run replay-reproducible.

## What the sandbox does **NOT** guarantee

- **Timing side channels.** A mod can measure its own execution (loop counts) and
  infer coarse timing; the sandbox does not defend against covert timing
  channels. Mods are semi-trusted content, not adversaries with a security
  boundary to breach for secrets.
- **Output-rate abuse within budget.** A mod that spams `log` or `spawn_cube`
  *within* its fuel budget is wasteful but contained (spawn ids are bounded by
  fuel); per-capability rate limits are a possible future hardening, not v1.
- **Semantic correctness.** Capabilities gate *reach*, not *intent*: a mod with
  `entities` can move any entity it can name (ids `1..N` in `Guid` order). Finer
  per-entity ACLs are a follow-up.
- **Floating-point cross-platform bit-identity of mod logic.** The engine's own
  sim is bit-reproducible (rapier `enhanced-determinism`); a mod's own f64 math
  is standard wasm f64 and is reproducible on a given target, but cross-target
  bit-identity of arbitrary mod arithmetic is not asserted.
- **Supply-chain trust of the mod author.** See below.

## Trust model

- **Mods are semi-trusted content**, treated like downloaded levels/assets: they
  run in-process but *sandboxed*, so a malicious or buggy mod cannot corrupt host
  memory, escape its capabilities, or hang the frame — the worst it can do is
  misbehave *within* its grants (move entities oddly, spam within budget) or trap
  (and get disabled). This is strictly stronger than the **dylib** hot-reload tier
  (`inf-hotreload`), which runs plugin code with full process trust over a
  `#[repr(C)]` vtable and is for first-party iteration only.
- **The grant is the policy.** Because `mod.toml` is out-of-band data an
  auditor/curator reads (not something the binary requests), a store/curator can
  vet the capability set before distribution. A mod cannot widen its own grant.
- **The wasm32 browser player loads no native mods** — `inf-wasm-host` (wasmtime)
  is native-only and gated off `wasm32`, so the mod tier never ships into the
  browser build.

## The cook flow (author → ship)

```
Blueprint (.inf_act)                     hand-written Rust (samples/mods/spinner)
        │  inf-transpile (SAME codegen                     │  inf-mod shim
        │  as the compiled/dylib path — parity)            │
        ▼                                                  ▼
  generated cdylib crate  ──cargo build --target wasm32──►  .wasm  ──►  mod.toml
        │                                                            (capability grant)
        └────────────────────────  inf cook --mods  ────────────────────┘
                                          │
                    inf-wasm-host sandbox (caps + fuel + memory)
                                          │
               inf-player --mods <dir>  /  editor Simulate (hot-reload)
```

- **Blueprint path.** `inf cook --mods <class.inf_act>` transpiles the class's
  event handlers through the existing transpiler, wraps them in the mod template
  (a `cdylib` + a host shim mapping the mod-host namespace onto the `inf-mod`
  ABI + the `mod_update` entry), and builds to `wasm32-unknown-unknown` when the
  toolchain is present (else honest instructions). v1 lowers the mod-host
  namespace subset (`host.*`, `input.*`); bridging the *entire* engine node kit
  (physics/audio/`vars`) onto wasm imports is the documented follow-up.
- **Rust path.** Author directly against `inf-mod` (the `samples/mods/spinner`
  shape). The committed spinner proves the full author→wasm→sandbox→sim story in
  a test (`inf-wasm-host` `spinner_e2e`, `inf-player` `mods_e2e`).

## Deferred / follow-ups (honest)

- Full engine node-kit → wasm-import lowering (v1 covers the mod-host subset).
- Wiring `ModsSession::{poll_reload, tick}` into `SimSession::fixed_step` behind
  an `EcsWorld` adapter (the reload mechanism + sandbox path are proven; the
  editor-Simulate glue is the remaining step — analogous to the player's
  `RuntimeSim::tick_mods`, already wired).
- Cross-reload mod-state migration (today a reload resets a mod's linear memory,
  like a fresh load; the dylib tier's schema-migration analogue is the follow-up).
- Per-capability rate limits and finer per-entity ACLs.
- Cook-time capability inference (today the generated `mod.toml` is a conservative
  default the author widens).
