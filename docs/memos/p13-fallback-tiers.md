# P13.4 — Classic-LOD fallback & GPU-capability auto-tier

*Phase 13 close. The virtualized-geometry meshlet path (P13.1) is the flagship;
this memo documents the **fallback** the roadmap requires ("classic-LOD fallback
documented for older GPUs") and the **capability detection + auto-tier** that
picks between them.*

## The three render tiers

Detected once at renderer/host init from the adapter, or forced via
`RenderSettings.tier_override`. A tier only ever **clamps features down** (via
`RenderTier::apply`) — it never turns a feature on, so the byte-stable defaults and
existing goldens are unaffected.

| Tier | Geometry path | Lighting / post | Chosen when |
|------|---------------|-----------------|-------------|
| **High** | GPU-driven **meshlet** path (`VgeomNode`): cull+LOD compute → vertex-pulled indirect draw | all requested effects (bloom, SSAO, TAA, shadows, GI) | adapter supports the meshlet path (compute + indirect + storage-buffer headroom) |
| **Medium** | **classic discrete-LOD** fallback (`ClassicVgeomNode`): the same content drawn through the ordinary PBR mesh pipeline | full lighting/post | can raster with storage buffers but lacks the meshlet-path headroom |
| **Low** | classic discrete-LOD | **effects off** (bloom / SSAO / TAA / shadows / GI disabled) | downlevel / software rasterizer / no storage buffers |

`RenderTier::apply` is the exact rule: **Medium** sets `vgeom.enabled = false`;
**Low** additionally clears `bloom`/`ssao`/`taa`/`shadows`/`gi`.

## What "classic-LOD fallback" means (true parity, not LOD0-only)

The classic path renders the **same** scene content as the meshlet path
(`RenderScene.vgeom_assets` + `vgeom_instances`) — `VgeomNode` runs only when
`vgeom.enabled`, `ClassicVgeomNode` only when it is off, so exactly one draws it and
scenes without vgeom content stay byte-identical.

The discrete LOD chain is **derived from the meshlet DAG's levels**
(`VgeomMesh::classic_lods()`): one index buffer per LOD level over the *shared*
vmesh vertex buffer (no vertex duplication). Per instance, the screen-space error is
projected to the **same** per-instance object-space threshold the meshlet cut uses
(`lod_threshold`), and the coarsest level within tolerance is picked
(`pick_classic_level`). So a far/small instance draws a coarse level and a near one
draws the finest — the classic twin of the meshlet cut, keyed off the *same* errors.
This is true classic-LOD parity, not a v1 LOD0 + distance-cull stand-in. Shadows /
GI / SSAO come along for free because the classic node reuses `mesh.wgsl`'s shared
`@group(2)` environment bind.

Measured on the gate scene (10.6M source triangles, 324 instances of one dense mesh):
a near camera draws ~300k triangles at max LOD 6; a far camera (whole grid tiny)
draws ~127k at max LOD 7 — coarser and cheaper, as intended.

## Capability detection signals

`AdapterCaps::probe` reads a portable subset of the adapter's limits/features/
downlevel flags (`caps.rs`):

- `compute_shaders` — downlevel `COMPUTE_SHADERS` (the cull compute needs it).
- `indirect_execution` — downlevel `INDIRECT_EXECUTION` (the vertex-pulled
  `draw_indirect` needs it).
- `max_storage_buffers_per_shader_stage` — the meshlet **raster** group binds 6
  storage buffers in one stage + 4 in the cull compute, so High requires ≥ 8
  (`VGEOM_MIN_STORAGE_BUFFERS_PER_STAGE`, the wgpu default).
- `max_storage_buffer_binding_size` — ≥ 128 MiB so a large meshlet/vertex payload
  fits one binding.
- `max_compute_workgroups_per_dimension` — ≥ 65 535 for the cull dispatch.
- `is_cpu` — a software rasterizer (WARP/lavapipe) is never High even if the limits
  nominally qualify (the meshlet path would be unusably slow).

`choose_tier` is a **pure** function of `AdapterCaps` (unit-tested on synthetic
capability sets, no GPU): **High** iff `supports_vgeom()`; else **Medium** iff it can
still raster with ≥ 1 storage buffer and is not a CPU adapter; else **Low**.

## The override

`RenderSettings.tier_override: Option<RenderTier>` bypasses detection:
`Some(tier)` forces it (the gate forces `Low` to prove the vgeom auto-disable),
`None` probes the adapter via `detect_tier`, which logs the decision. Because
`apply` only clamps down, an override can only *reduce* capability — it cannot force
a GPU to attempt a path it cannot run.

## Where it lives

- `crates/inf-render/src/caps.rs` — `AdapterCaps`, `RenderTier`, `choose_tier`,
  `detect_tier`, `RenderTier::apply`.
- `crates/inf-render/src/passes/classic_vgeom.rs` — `ClassicVgeomNode` +
  `classic_lod_selection` (the CI-provable draw/instance probe).
- `crates/inf-vgeom/src/model.rs` — `VgeomMesh::classic_lods` / `classic_lod_errors`
  / `pick_classic_lod` + `ClassicLod`.
- `runtime/inf-player/src/render.rs` — the player probes the tier at host init,
  resolves `MeshRef.asset` → vgeom content, and renders through the picked path.
- Gate: `runtime/inf-player/tests/vgeom_gate.rs`.

## Deferred (documented follow-ups)

- **PIE** streams no vmesh index yet, so asset meshes render as placeholder cubes in
  play-in-editor until the `ScenePayload` carries the vmesh set.
- The **editor viewport** still draws the `MeshRef.primitive` placeholder for asset
  meshes (the asset-DB-in-viewport binding is the standing Phase-4→7 follow-up); the
  **player** renders the real geometry.
- Per-project **persistence** of the tier override / a manual tier selector in the UI.
- Storing the classic LOD index buffers **in the `.inf_vmesh` payload** at cook time
  (they are currently derived from the levels at load — cheap and lossless, so this
  is an optional size/latency trade, not a correctness gap).
