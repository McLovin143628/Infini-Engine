# Memo: console-readiness HAL audit (P14.4)

**Status:** audit complete — the seam is defined and proven by a null backend; the
engine does **not yet route through it** (honest gap table below).
**Date:** 2026-07-21
**Scope:** whether every OS/GPU/file/input/time dependency a shipped game touches
can be isolated behind `inf-platform` traits, verified by a mock "null console"
backend; the cook-target plumbing and private-repo pattern for out-of-tree console
backends; TRC/TCR checklist drafts.

## What was built this phase

- **`inf-platform`** — the HAL seam, now real (it was a `//!` doc-comment stub from
  P0). `std`-only, no third-party deps: traits `Clock`, `FileSystem`,
  `SaveDataStore`, `ThreadSpawner`, `DisplayTarget`, `InputBackend`, plus a
  `LifecycleSink` (TRC suspend/resume) and an assembled `Hal` handle. A `desktop`
  module provides the `std`-backed real backend.
- **`platforms/inf-platform-null`** — a complete mock backend: in-memory VFS,
  deterministic fake monotonic clock (advances only when told), inline
  (deterministic) thread spawner, headless display, no-device input, in-memory
  save slots. Its test instantiates the `Hal` and exercises **every** required
  seam — the proof that a backend with no OS, no GPU, and no devices satisfies the
  contract, so a real console backend is additive.

## The honest finding

Standing up the seam is not the same as the engine using it. **Today almost no
Ring-0 crate consumes `inf-platform`** — the HAL was a P0 stub, and subsystems
built since reach for `std` and `wgpu`/`winit` directly. The audit's real output is
therefore a **gap table**: where the engine bypasses the seam, and the concrete
seam work each gap needs. This is the bounded, additive work a console port
front-loads — exactly what §9 promises ("a bounded, additive project rather than a
refactor"), stated with the current bypasses named rather than hand-waved.

### Gap table (Ring-0 `crates/`)

| Seam | Trait | Consumers today | Behind HAL? | Gap / seam work |
|---|---|---|---|---|
| Files (game data) | `FileSystem` | `inf-asset` (db, pack, import_cache, sidecar), `inf-scene`, `inf-project`, `inf-graph` (cache), `inf-mesh` (glTF), `inf-audio` (mixer file loads) | **No** — direct `std::fs` | Route asset/level/pack reads through a `FileSystem` handle; consoles mount read-only title storage + a separate writable area. Highest-volume gap; mechanical but broad. |
| Save data | `SaveDataStore` | none yet (editor autosave is Ring-1; runtime has no save system) | n/a | New system; must be `SaveDataStore` from day one (console save-data is slot/quota/async, **not** a file path). No retrofit debt if built on the seam now. |
| Time | `Clock` | `inf-render` (surface resize debounce, `Instant`), `inf-physics` (timing) | **No** — direct `Instant`/`SystemTime` | Non-critical: the **sim already takes no wall clock** (caller-fed deltas — §2.5), so the determinism-sensitive path is clean. Frame-pacing/debounce reads move to `Clock`. |
| Threads | `ThreadSpawner` | none in Ring 0 (compute goes through `inf-core`'s rayon job pool; only `inf-platform::desktop` spawns) | **Effectively yes** | Best-shape gap: gameplay/asset parallelism is already funneled through `inf-core` (one pool to size/pin per platform). Long-lived IO threads (asset watcher, audio, net pump) should take a `ThreadSpawner`. |
| GPU / window | `DisplayTarget` | `inf-render`, `inf-render-2d` (pervasive `wgpu`), viewport/`winit` in Ring 2 | **No** — `wgpu` directly | Largest *conceptual* gap. `DisplayTarget` pins backend-neutral geometry + present policy; the renderer must accept a display/surface handle instead of a concrete window. Console GPU is a wgpu-backend/native-API question the seam brackets but does not solve. |
| Input | `InputBackend` | `inf-input` references `winit` types (touch/event conversion); `gilrs` behind a feature | **Partial** | `inf-input`'s action/axis **core is already backend-neutral**; the gap is the `winit`/`gilrs` event *source*. Feed `InputEvent`s from the platform backend into the existing mapper. |

**Net:** the two seams that matter most for determinism/portability are already in
good shape — **compute threading** (funneled through `inf-core`) and the **sim's
clock independence** (§2.5). The broad-but-mechanical gap is **file IO**; the deep
gap is the **GPU/window** seam (inherent to any engine — bracketed here, not
pretended solved). Save-data is greenfield and must be built *on* the seam.

## Cook-target plumbing (P14.4 item 2)

A console build differs from desktop in three places, all additive:

1. **Target triple + backend crate.** The cook selects a platform backend crate
   (e.g. `inf-platform-<console>`) the same way a test selects
   `inf-platform-null` — it is *not* a dependency of any engine crate; the top-level
   binary picks one. The cook records the chosen `platform` in the pack/boot config
   so the player knows which `Hal` to assemble.
2. **Asset conditioning.** Texture formats, audio codecs, and shader variants are
   platform-conditioned at cook time (the pack is content-addressed already —
   P9.2 — so per-target packs are just a different input set).
3. **Toolchain.** The console SDK/toolchain lives in the **private** workspace
   (below), invoked by the cook there; nothing SDK-specific enters this repo.

The seam already present: `inf cook`/pack is content-addressed and deterministic,
and the runtime reads the pack through `inf-scene` in byte-lockstep with the editor
codec (P9.2). Adding a `--target <platform>` that flips the backend crate + asset
conditioning is the remaining plumbing (a bounded CLI/packager change owned by the
concurrent packager work — flagged, not done here).

## The private out-of-tree backend pattern (P14.4 item 3)

Console SDKs are NDA'd and cannot live in this public repo (risk register #5). The
pattern, **demonstrated** by `platforms/inf-platform-null`:

```
# public repo (this one)
crates/inf-platform/                 # the seam (traits + std desktop backend)
platforms/inf-platform-null/         # in-repo mock backend — proves the seam

# a SEPARATE private repo (per platform holder, devkit + NDA SDK)
platforms/inf-platform-ps5/          # implements the same inf-platform traits
platforms/inf-platform-xbox/         #   against the private SDK
platforms/inf-platform-switch/       #
```

Rules that make this hold:

- **Nothing in Ring 0 names a backend.** The engine depends only on
  `inf-platform` traits; the concrete `Hal` is assembled at the top level (player
  binary / cook target). `inf-platform-null` is deliberately **not** a dependency
  of any engine crate — a console backend is selected the identical way.
- **`platforms/` sits outside `crates/`** and outside the engine dependency graph,
  signalling "selected, not depended-on".
- The private repo is an overlay: it adds `platforms/inf-platform-<console>/` and a
  player/cook target that selects it, pinning this repo as a dependency. No engine
  fork.
- CI in the public repo builds `inf-platform-null` and runs its seam-coverage test,
  so seam **regressions** (a new required capability with no null impl) are caught
  here without any SDK.

## TRC/TCR compliance checklist drafts (P14.4 item 4)

Generic, no NDA content — the *categories* every console certification covers, as a
drafting checklist. Concrete per-platform requirement text lives only in the
private repo.

**Controller / input**
- [ ] Correct platform button **glyphs** shown in all UI (never a foreign platform's
      names). Seam: `InputBackend` reports the platform; a glyph set keys off it.
- [ ] Controller disconnect → pause + reconnect prompt; graceful re-pair.
- [ ] Support the platform's required controller counts / hot-join.

**Lifecycle (suspend / resume / constrained)**
- [ ] Suspend within the platform's deadline; resume restores state seamlessly.
      Seam: `LifecycleSink::on_lifecycle` (checkpoint save, pause audio, release GPU).
- [ ] Constrained/background state reduces resource use as required.
- [ ] Clean quit within deadline (flush saves, close handles).

**Save data**
- [ ] Use the platform save-data API (slots/quota), **not** raw file writes.
      Seam: `SaveDataStore`.
- [ ] Corruption-tolerant (atomic write, validate on load); handle "storage full".
- [ ] No blocking of the main thread on save IO; progress/blocking-dialog rules met.

**Performance / stability**
- [ ] Meet the platform's minimum sustained frame-rate and load-time bars.
- [ ] No crashes/hangs in cert scenarios; error dialogs use platform UX.

**Content / presentation**
- [ ] Safe-area / overscan honored; required resolutions & HDR handling.
- [ ] Age-rating, legal, and account/online-status messaging per platform rules.

**Online (if applicable)**
- [ ] Use platform networking/session/matchmaking services as required.
- [ ] Privacy/parental-control and communication-restriction settings respected.

## Decision / outcome

- The console seam is **defined and provably satisfiable** (null backend, all seams
  green in CI).
- The engine's **actual** platform coupling is documented, not hidden: the
  determinism-critical seams (compute threading, sim clock) are already clean; file
  IO is the broad mechanical gap; GPU/window is the deep gap inherent to any engine;
  save-data is greenfield and must be built on the seam.
- A console port is therefore the bounded, additive project §9 promises: implement
  the traits in a private crate, and close the named gaps (route file IO through
  `FileSystem`, accept a `DisplayTarget` in the renderer, source input events into
  the existing mapper). None of it is a rewrite.

## Follow-ups (owned, not done here)

- Route `inf-asset`/`inf-scene`/`inf-project` file IO through a `FileSystem` handle.
- Have `inf-render` accept a `DisplayTarget`/surface handle instead of a concrete
  window (coordinate with the renderer owner).
- Add a `SaveDataStore`-based runtime save system.
- `inf cook --target <platform>` backend selection + asset conditioning (packager).
