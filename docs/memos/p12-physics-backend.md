# P12 memo — Physics backend: rapier vs. Jolt

**Status:** decided · **Date:** 2026-07-21 · **Phase:** P12.1/12.2
**Verdict:** stay on **rapier** (`rapier2d-f64` / `rapier3d-f64` 0.34) through P14, with
named revisit triggers. Revisit is a *port*, not a rewrite, because the whole engine only
ever speaks the `inf-physics` facade.

---

## Why this memo exists

P12 deepened physics (joints, filtering, materials, CCD, ragdolls) and made determinism a
shipped, tested guarantee (the P12.2 replay gate). That is the moment to write down, on the
record, whether the substrate under the `inf-physics` facade is the right one before we build
gameplay systems on top of it — and to fix the criteria that would make us change our mind.

The only two credible options for a Rust engine today are **rapier** (pure Rust) and **Jolt**
(C++, via FFI). This is a grounded comparison of the two, not a survey.

## What we have, and why it insulates the decision

Every physics type the engine touches is an `inf-physics` facade type — `PhysicsWorld{2,3}D`,
`BodyId3D`, `ColliderDesc3D`, `JointDesc3D`, the ECS bridges. **No crate outside `inf-physics`
names rapier, parry, or nalgebra.** (Architecture rule 1 + the crate's own module docs enforce
this; the bridges consume plain snapshots.) The backend is therefore a swappable implementation
detail: switching engines is re-implementing one facade crate against a different library, with
the determinism harness (`tests/playground_determinism.rs`) as the acceptance test. That single
fact lowers the stakes of choosing "wrong" from *engine rewrite* to *one-crate port*, and it is
why we can commit to rapier now without betting the project on it.

## rapier — the incumbent

**For**

- **Pure Rust, one dependency tree.** No C++ toolchain, no `cc`/CMake/ISPC in CI, no
  cross-compilation of a native lib for three desktop OSes (we already paid the ISPC tax once
  with `intel_tex_2` in P4 and deliberately backed it out — see the BC7 deferral). rapier builds
  wherever `cargo` does.
- **f64 build.** `rapier{2,3}d-f64` simulates in double precision, which is *load-bearing* for
  us: architecture rule 3 makes world-space f64 with floating-origin rebasing. A physics engine
  that ran in f32 would re-introduce exactly the precision cliff floating-origin exists to avoid
  (an f32 coordinate jitters past a millimetre ~4 km out). Jolt is f32-only. This is the single
  biggest differentiator and it points at rapier.
- **glam-native.** rapier 0.34's `-f64` build resolves its math types to glam 0.33 (rapier →
  parry → glamx → glam), our exact workspace pin. A rapier `Vector` *is* a `DVec3`; the facade's
  conversion is the identity. No second glam, no marshalling layer.
- **`enhanced-determinism` is a first-class, contractual feature.** It routes transcendentals
  through `libm` (simba `libm_force`) for IEEE-754 cross-platform reproducibility. Our P12.2 gate
  (300 steps × 2 runs, byte-identical poses, 2D *and* 3D, composing stacks + a joint zoo +
  motors + CCD + sensors + layer filtering) passes on this. Determinism is the product promise
  behind replays, netcode, and the transpiler parity story — rapier gives it to us by
  construction.
- **API breadth already covers P12.** Impulse joints (fixed/revolute/prismatic/spherical/rope)
  with motors + limits, `InteractionGroups` filtering, `CoefficientCombineRule`, per-body CCD —
  all shipped this batch against rapier's own types, no gaps hit.
- **Same author as parry** (our collision/query lib) **and nalgebra**; the ecosystem is
  coherent and actively maintained.

**Against**

- **Solver maturity / perf headroom.** rapier's constraint solver is good but younger than
  Jolt's; on very large or very stiff scenes (thousands of active contacts, tall stacks, dense
  ragdoll piles) Jolt's multithreaded island solver has a real, measured lead in the wider game
  community. We have not hit this wall — but we also have not shipped a physics-heavy scene.
- **Single-threaded by our own choice.** We keep rapier's `parallel` (rayon) feature *off* for
  determinism and to avoid a second worker pool beside `inf-core`'s job system. That caps
  per-world throughput at one core. Our intended scaling axis is *many independent worlds across
  the deterministic ECS schedule*, not one giant parallel world — but if a single world must be
  huge, this is a constraint.

## Jolt — the challenger

**For**

- **Console/AAA pedigree.** Jolt ships in Horizon Forbidden West; it is proven at shipping-title
  scale and on console hardware. Its multithreaded solver and large-scene stability are its
  headline strengths.
- **Performance headroom** on the hardest scenes, per the same community benchmarks.

**Against**

- **C++ + FFI.** Consuming Jolt means either `jolt-rust`/`joltc` bindings or maintaining our own,
  plus a C++ build in CI on Windows/macOS/Linux and for every console target. That is precisely
  the cross-OS native-build liability we have twice chosen to avoid (git2 → git CLI in P5; ISPC
  BC7 deferral in P4). FFI also complicates the hot-reload story and `catch_unwind` isolation.
- **f32 only.** No f64 build exists. Marrying Jolt to our f64 world means either simulating in a
  rebased f32 frame around the floating origin (extra machinery, per-frame rebasing of the whole
  physics world, and a determinism surface to re-audit) or abandoning rule 3 for physics. Both
  are real costs that rapier simply does not impose.
- **Determinism is not the same contract.** Jolt documents cross-run determinism on the *same*
  binary/platform but is explicit that it is **not** guaranteed cross-platform. Our promise is
  cross-machine byte-identity; matching it on Jolt would be additional, ongoing verification work
  that rapier's `enhanced-determinism` gives us as a feature flag.
- **Ecosystem seam.** Jolt has its own math + shape types; the glam-identity we enjoy with
  `rapier-f64` disappears and the facade grows a real conversion layer.

## Decision criteria (the trade, stated plainly)

| Axis | Weight for us | Winner |
|---|---|---|
| f64 world / floating-origin fit | **critical** (rule 3) | **rapier** |
| Cross-platform determinism (product promise) | **critical** | **rapier** |
| CI / build simplicity (pure Rust, no FFI, console builds) | high | **rapier** |
| glam-native integration | medium | **rapier** |
| Single-world solver perf on huge/stiff scenes | medium | Jolt |
| Console cert pedigree | medium (later) | Jolt |
| API breadth for P12 features | table stakes | tie (both have them) |

Everything we weight *critical* or *high* points at rapier, and two of them (f64, cross-platform
determinism) are things Jolt structurally **cannot** match without significant added machinery.
Jolt wins on perf headroom and console pedigree — both of which are (a) not yet needs we have
measured, and (b) recoverable later via the facade port.

## Verdict

**Stay on rapier through P14.** Build gameplay physics (character controllers, vehicles,
ragdoll runtime, destruction) on the `inf-physics` facade as-is. Keep the facade the only place
that names a physics library, and keep the P12.2 determinism harness green as the contract the
backend must satisfy.

### Revisit triggers (any one flips this to an evaluation spike)

1. **A perf wall on a real budget.** A shipping-representative scene misses its frame budget in
   the physics step *specifically*, and profiling shows the rapier solver — not our sync/bridge —
   as the cost, after we have already tried the many-independent-worlds scaling axis. "It might be
   slower" is not a trigger; a missed budget on a real scene is.
2. **Console certification needs.** When a console port becomes concrete and either its cert
   requirements or its perf targets specifically favour Jolt's proven console solver. (Evaluate
   the FFI/build cost against the determinism regression at that point, with real hardware.)
3. **Determinism regression we cannot fix in rapier.** If `enhanced-determinism` ever fails a
   platform we must ship (and upstream cannot fix it), re-evaluate — though this would count
   *against* Jolt too, whose cross-platform determinism is weaker.

If a trigger fires, the work is: implement `inf-physics` against Jolt behind the existing facade,
make `tests/playground_determinism.rs` and the per-facade suites pass (or consciously renegotiate
the determinism contract), and swap. The rest of the engine does not change. That is the whole
point of the facade — and the reason committing to rapier now is low-risk.
