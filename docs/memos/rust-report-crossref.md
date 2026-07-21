# Memo — Cross-reference of the "Rust vs C++ in Modern Game Development" report against the roadmap

**Status:** DONE (2026-07-20). A 4-page external technical report (`../docs/rust_game_dev_report.pdf`,
outside the repo) was used as a best-practices checklist against `docs/ROADMAP.md`. Verdict: the
plan already covers essentially the whole report — two amendments were warranted; both are now in
the roadmap. This memo records the findings and the decisions so they are not relitigated.

## What the report recommended (Section 6) vs. what we had

| Report item | Roadmap state at review | Action |
|---|---|---|
| Modular layered architecture | Three-ring rule (§2) | none — covered |
| Drag-and-drop 3D viewport | Phase 2, **built + CI-green** | none — done |
| Entity picking (GPU ID buffer / raycast) | **Already best-practice** — `crates/inf-render/src/pick.rs` renders every instance to an `R32Uint` id target and reads back the texel under the cursor; wired to selection via `inf-viewport` `pick_guid`. Gizmo handles are a separate analytic hit-test in `gizmo.rs`. | none — done, better than report |
| Visual material graphs → WGSL transpiler | Phase 7 (`.inf_mat` → naga-validated WGSL), owned P7.2 | none — planned |
| Animation blend trees / state machines | Phase 11, owned P11.2 (`.inf_sm`) | none — planned |
| Hot reload (dylib) | Spike C + `inf-hotreload`, **built** | none — done |
| Pure ECS, cache-friendly | Principle 3, `inf-ecs` over bevy_ecs | none — covered |
| **Parallel systems / fearless concurrency** | Unbuilt facade, no owning phase | **Amendment 1** |
| Console SDK / NDA / FFI barrier (Section 3) | Principle 6 + §9 + P14.4 — HAL seam, null-console mock, private out-of-tree crates | none — covered, *more* rigorous than the report |
| Embedded scripting VM (Rhai/Rune/WASM) as a hot-reload alt | Not considered | **Amendment 2** (WASM, scoped) |
| rust-gpu shaders | WGSL + naga chosen | none — decided |

## Amendment 1 — multi-threading is now a first-class, owned concern

Findings (verified in code): `crates/inf-core/src/lib.rs` was a one-line doc stub with empty
`[dependencies]`; `inf-ecs` used `bevy_ecs` as a **data store only** (no `Schedule`,
`multi_threaded` not enabled); `inf-runtime/src/sim.rs` is a serial loop; no `rayon`/`flume`/
`num_cpus` appeared in any `Cargo.toml` (nor in `Cargo.lock`). Concurrency existed only as
architecture asides plus one late Phase-15 "parallel import/cook tuning" line — no phase owned it.

Roadmap changes: new **§2.5 Concurrency & parallelism model** (doctrine: rayon compute pool +
`flume` in `inf-core`; tokio confined to Ring 2; parallel `bevy_ecs` `Schedule`; parallel-yet-
deterministic fixed step, replay-tested); a concrete job-system deliverable **P7.0**; an explicit
parallel-ECS sub-item in **P9.1**; named hot loops per owning phase; a **§8 CI concurrency-
determinism gate** (parallel schedule must produce a byte-identical trace to the serial baseline);
cross-references fixed on the `inf-core` layout line and the tech-matrix Concurrency row.

## Amendment 2 — a WASM extensibility/mod tier, not a scripting language

Decision: **do not** add a separate embedded scripting language (Rhai/Lua/Rune). The blueprint
tree-walking interpreter already provides no-recompile *iteration* over the same IR that ships as
Rust, so another language would add a fourth execution model and fracture product principle 2
("two ways to code, one truth"). The one capability the design genuinely lacked is **safe, no-
compiler, sandboxed end-user modding / runtime plugins** — dylib mods are ABI-fragile and unsafe;
blueprints are not runtime-user-facing. That is served by **P14.5**: a sandboxed **WASM** runtime
(`wasmtime`) loading mods compiled from the *same* Blueprints/Rust (blueprint → Rust → wasm) via a
WASM cook target, behind a capability-scoped host API. This preserves "one truth" and resolves the
pre-existing P14.2 "interpreter vs wasm-compiled blueprints" decision toward one shared WASM path.

## Deliberate non-goals recorded

Separate embedded scripting language (rejected, see above); WGSL over `rust-gpu`; native wgpu
viewport over egui (§2.3.1). Console/FFI is not a gap — it is honestly deferred to P14.4 behind the
`inf-platform` HAL seam with a null-console mock proving the seam.

## Scope of this task

Roadmap + this memo only — no engine code changed. Building the `inf-core` job system + parallel
ECS happens when P7.0 / P9.1 are reached; the WASM tier at P14.5.
