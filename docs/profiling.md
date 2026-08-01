# Profiling, performance budgets & stability (P15.1 / P15.2)

This document covers how to profile Infinity Engine with Tracy, the performance
budgets enforced in CI, the memory diagnostics, and the crash-reporting /
autosave-recovery machinery. It is the reference for the P15.1 (performance) and
P15.2 (stability) work.

## 1. Tracy profiling

The engine emits `tracing` spans on its hot paths **unconditionally** — they are
part of the normal instrumentation. Recording them into the
[Tracy](https://github.com/wolfpld/tracy) live profiler is done by adding a
subscriber *layer*, gated behind an off-by-default `tracy` cargo feature. So:

- A default build has **zero** profiling overhead and never compiles the vendored
  C++ Tracy client.
- Turning on `tracy` adds the `tracing-tracy` layer; the spans that already exist
  stream to a connected Tracy viewer.

### Running with Tracy

Editor (Studio):

```sh
cd editor/studio
cargo tauri dev --features tracy
# or a plain backend build:
cargo build -p inf-studio --features tracy
```

Standalone player / PIE subprocess:

```sh
cargo run -p inf-player --features tracy -- --pack <build-dir>
```

Then launch the Tracy **profiler** GUI (a separate download from the Tracy
project) and connect — spans stream live.

### What is instrumented

| Span | Level | Where | Crate |
|------|-------|-------|-------|
| `render_frame` | info | `EngineRenderer::render` (once per frame) | `inf-render` |
| `render_node` (per pass, `name=…`) | trace | `RenderGraph::run` | `inf-render` |
| `sim_step` | info | `GameLoop::step_once` (once per fixed step) | `inf-runtime` |
| per-phase sim systems (`attract_to_centroid`, `integrate_velocity`, `age_lifetime`, `apply_spawns`, `propagate_transforms`) | trace | `inf_ecs::sim` | `inf-ecs` |
| `cook`, `cook_read`, `cook_assets`, `derive_vmesh` | info | `inf_packager::cook` | `inf-packager` |
| `import_textures` | info | glTF image decode (parallel) | `inf-editor-core` |
| `build_vgeom` | info | meshlet-DAG build | `inf-vgeom` |

The `info`-level spans show up under Tracy at the default filter. The finer
per-pass / per-phase spans are `trace`-level (so they cost nothing at the default
`info` filter); to see them, raise the filter for those targets, e.g.:

```sh
RUST_LOG=inf_render=trace,inf_ecs=trace cargo run -p inf-player --features tracy -- …
```

> **Doctrine (§2.5 / tech matrix):** spans exist in the code regardless of the
> `tracy` feature; only the *subscriber layer* is feature-gated. Do not wrap
> spans themselves in `#[cfg(feature = "tracy")]`.

## 2. Performance budgets (the §8 ratchet)

CI runs hard, generous frame-budget smokes as ordinary tests (they are fast, so
they fold into the normal `cargo nextest` job rather than a separate nightly
job). Each budget is a **committed constant with a ratchet rule: it may only ever
be lowered, never raised.** A regression must be fixed, not accommodated.

| Budget | Constant | Value | Test |
|--------|----------|-------|------|
| Sim fixed step (mean, ~275-entity replay world) | `SIM_STEP_BUDGET_MS` | 2.0 ms | `inf-runtime` · `tests/sim_budget.rs` |
| Render frame (mean, ~484-cube scene, full pipeline) | `FRAME_BUDGET_MS` | 33.0 ms | `inf-render` · `tests/frame_budget.rs` |
| Editor project-open (asset scan) | `OPEN_BUDGET_MS` | 5000 ms | `inf-editor-core` · `tests/startup_budget.rs` |
| Player pack-load-to-first-world | `STARTUP_BUDGET_MS` | 5000 ms | `inf-player` · `tests/startup_budget.rs` |
| **Streamed** fixed step (mean, cell + terrain streaming live) | `STREAMED_STEP_BUDGET_MS` | 4.0 ms | `inf-player` · `tests/phase16_gate.rs` |
| Terrain page bytes resident (peak over the flythrough) | `TERRAIN_RESIDENT_BYTES_CEILING` | 16 MiB | `inf-player` · `tests/phase16_gate.rs` |
| Partition cell bytes resident (peak) | `CELL_RESIDENT_BYTES_CEILING` | 256 KiB | `inf-player` · `tests/phase16_gate.rs` |
| Partition cells active at once (peak) | `CELL_RESIDENT_CEILING` | 8 | `inf-player` · `tests/phase16_gate.rs` |

Notes:

- The four **streamed-scene** budgets (P16.6) live in `inf_player::budget`, are
  asserted over the composed `samples/phase16-world` gate scene, and print their
  measured values on every run (step 0.18 ms, terrain 5.65 MiB, cells 2.8 KiB / 4
  active on a developer machine) — read the line, then lower the constant. Their
  module docs also state, honestly, what a byte ceiling on a gate-sized scene can
  and cannot catch, and why the **120 fps-class frame-rate claim itself stays
  human-verified on real hardware**: CI can bound the CPU-side streaming work, not
  a frame.

- **`frame_budget`** skips when no GPU adapter is present, and on a **software**
  adapter (llvmpipe/WARP on CI) it only *smoke-renders* — the strict budget is
  enforced only on real hardware, where the number is meaningful (mirrors the
  golden-image harness philosophy). The values above are generous floors; lower
  them once the measured hardware floor is known.
- The `sim_budget` / `startup_budget` tests need no GPU and run on every OS.
- The criterion benches (`inf-core/benches/job_pool`, `inf-runtime/benches/schedule`)
  remain the fine-grained scaling measurements; the budget tests are the
  pass/fail tripwires.

## 3. Parallel cook

The cook pipeline's per-asset CPU work — scene decode/re-encode, blueprint
decode+validate, and meshlet-DAG (`.inf_vmesh`) derivation — is fanned across the
Ring-0 job pool via `inf_core::parallel_map` (a **deterministic, in-order** map).
The results are folded back into the pack **serially, in closure order**, and the
`PackWriter` stores by GUID, so the cooked pack is **byte-identical** regardless
of pool size — the P9.2 cook-determinism gate
(`inf-packager/tests/cook_platformer.rs`: `cook_is_deterministic`,
`cook_with_vmesh_derivation_is_deterministic`) still holds. The fold `?`-es on the
first error in closure order, preserving the fail-fast, handler-anchored
first-broken-blueprint contract. zstd compression stays a serial fold inside
`PackWriter::add_bytes` (parallelizing it would need a pre-compressed-blob writer
API — a documented follow-up); the largest CPU stage (the meshlet build) is now
parallel across meshes and internally parallel per mesh.

## 4. Memory diagnostics

`inf_editor_core::diagnostics::MemoryReport` estimates the editor's live memory
per subsystem: entity count, scene-document bytes, undo/redo depth, terrain
tiles + bytes, and (filled by a Ring-2 caller when the viewport is attached)
texture-cache / vgeom / audio bytes.

Backend surface (frontend adoption is a **documented handoff** — no frontend
changes here):

- Ring-2 command **`editor_diagnostics`** returns a `MemoryReport` for the active
  scene and logs a one-line `report.summary()` to the Output Log. A frontend
  `stats` command or status-bar readout can adopt it later; the renderer/runtime
  cache fields are the fields a future Ring-2 caller fills from the viewport
  thread.

## 5. Crash reporter (P15.2)

No telemetry is uploaded. Opt-in upload is a **config flag + docs only**, never
silent — see §7.

- **Player (`inf-player`):** the panic hook writes a structured crash file
  (`inf-player/src/log.rs`): engine version, OS/arch, GPU adapter, timestamp,
  panic location + message, and the last N log lines (a 256-line ring). It writes
  the primary `--crash-file` (default `crash.txt`) **and** a timestamped copy into
  a `crashes/` directory beside it, so repeated crashes each leave a record.
- **Editor (`inf-studio`):** a Rust panic hook (`commands/diagnostics.rs`,
  installed from `lib.rs`) writes a timestamped `crash-<millis>.txt` into the
  app-data `crashes/` dir with engine version, OS/arch, panic location + message,
  and a tail of recent log lines (from the Output-Log bridge ring). Chains to the
  default hook.
- **Viewport thread (`inf-viewport`):** the Win32 render loop wraps the frame in
  `catch_unwind`; a panic in engine render code writes an `inf-viewport` crash
  report (to a temp `crashes/` dir), logs a graceful message, and exits the render
  loop cleanly — the editor process and its webview survive instead of the whole
  app dying. (Complements the existing P2.1 device-lost recovery in
  `host.rs`/`gpu.rs`.)

The crash-report formatting + `write_crash_report` live in Ring-1
`inf_editor_core::diagnostics` (Tauri-free, unit-tested).

## 6. Autosave & recovery hardening (P15.2)

Autosave is a 5-second frontend interval that calls `scene_autosave`, which writes
a `crash-recovery.inf_lvl` **only when the document is dirty**; an explicit save
clears it. On boot, `recover_scene_on_boot` consumes any surviving recovery file.

Hardening in `inf_editor_core::scene::serialize`:

- `take_recovery` never panics on a **corrupt / truncated** recovery file — it
  moves the bad file aside to `crash-recovery.inf_lvl.corrupt`, logs a warning,
  and returns `None` so startup falls back cleanly to the last good save.

Tests (`serialize.rs`): `autosave_only_persists_a_dirty_doc`,
`recovery_restores_the_pre_crash_document`,
`corrupt_recovery_file_is_handled_gracefully`, plus the pre-existing
`recovery_round_trips_then_clears`.

## 7. Long-session soak & opt-in telemetry

- **Soak seed:** `inf-editor-core/tests/soak.rs` — an `#[ignore]`d, deterministic
  10 000-cycle edit/undo/redo/save workout that asserts invariants every step and
  that memory does **not** grow unboundedly (undo depth stays bounded by the
  history limit; entity count / scene bytes stay under a ceiling). Run manually /
  nightly:

  ```sh
  cargo test -p inf-editor-core --test soak -- --ignored --nocapture
  ```

- **Opt-in telemetry (stub):** there is deliberately **no** crash/telemetry
  upload implemented. If added, it must be an explicit opt-in config flag
  (default off) with a clear consent UX — never a silent background send. This is
  a documented non-goal for now, recorded here so the crash reporter is not
  mistaken for a telemetry channel.
