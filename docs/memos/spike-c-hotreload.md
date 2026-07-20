# Spike C memo — hot reload of game-logic dylibs

**Status:** GO. `inf-hotreload` loads plugin cdylibs through `#[repr(C)]`
fn-pointer vtables, carries live component state across a reload (including
a schema migration that defaults new fields), contains panicking plugin code
without harming the process, and dodges the Windows file lock. Verified by
an in-process suite plus an integration suite that builds and reloads two
real fixture cdylibs (`test-plugins/v1`, `v2`) on every CI run.

## Contract

- **No Rust types cross the boundary.** Only `#[repr(C)]` structs of fn
  pointers, raw byte pointers, `u32` status codes. `ABI_VERSION` is checked
  at load; a mismatch refuses the plugin.
- Components register as **(name, schema_hash, create, destroy, serialize,
  deserialize, tick)**. Name is the stable identity reload matches on.
- Buffers move through **host-owned callbacks** (`WriteFn` for bytes,
  `LogFn` for messages) — no allocation ever crosses the boundary, so there
  is no cross-module `free` and no allocator mismatch class of bug.

## Decisions

1. **Never unload.** Unloading a Rust dylib is UB-adjacent (TLS dtors,
   `'static` references into the image, vtables still referenced). Every
   `Library` is leaked; a reload just swaps which vtable instances point at.
   Old images stay mapped for the editor session — accepted memory cost.
2. **Content-addressed shadow copies** (`{stem}.{xxh3 of bytes}.{ext}` in a
   shadow dir) are what actually gets loaded. The build artifact on disk is
   never locked — the test overwrites it *while its copy is loaded* — so
   `cargo build` always succeeds, and re-loading unchanged bytes reuses the
   same shadow (dedupe for free).
3. **Panics are caught on the plugin side.** An unwinding panic through
   `extern "C"` aborts the process, so the host cannot defend itself — the
   guest shims (`inf_hotreload::guest`) wrap every entry in `catch_unwind`,
   log the panic message through the host's `LogFn`, and return
   `STATUS_PANICKED`. The host then **disables that one instance** (Output
   Log gets the message); everything else keeps ticking. A later successful
   reload re-enables it — reloading fixed code is the point.
4. **Reload = old serializes → new deserializes.** State crosses as
   self-describing JSON; fields the old schema lacked are filled by serde
   `#[serde(default)]` in the new dylib. `schema_hash` (xxh3 of a
   human-maintained descriptor string) mismatch is *reported* as a
   migration, never fatal. Failure to serialize/deserialize keeps the
   instance on the old (still-loaded) code — reload never loses state.
   JSON is the spike/tooling choice; if profiles show it hot, a tagged
   binary format can slot in behind the same vtable without ABI changes.
5. **Guest ergonomics:** plugins implement `HotComponent` (serde +
   `Default` + `tick`) and call `export_plugin!("version", [A, B])` once.
   The macro builds the leaked vtable in a `OnceLock` behind the single
   `infinity_plugin_entry` export. Generic `extern "C"` shims monomorphize
   per component — no proc macros, no codegen step.
6. **`abi_stable` rejected** (as anticipated in the ROADMAP tech matrix):
   the vtable surface is 7 fns; hand-rolling keeps the boundary auditable
   and dependency-free for user plugin crates
   (`inf-hotreload` with `default-features = false` is serde + serde_json +
   xxhash only).

## Verified (2026-07-19, in CI from this commit on)

- In-process suite (`tests/inprocess.rs`): spawn/tick/serialize round trip,
  panic disables only the broken instance with the message captured, ABI
  version mismatch and duplicate component names rejected.
- Dylib suite (`tests/reload.rs`) — builds both fixture plugins with cargo,
  then: v1 ticks state up; `fragile` panics on tick 3 and is contained;
  reload onto v2 carries `ticks: 5`, defaults the new `bonus: 1.5` field
  (reported as migrated), re-enables `fragile`, and v2 behavior (+10/tick)
  takes over the carried state. Content-addressed loading reuses shadows;
  the original dylib is overwritten while loaded (the Windows lock test).

## Open items / accepted losses

- The plugin's own panic hook prints the raw panic to stderr before the
  shim catches it — cosmetic noise; production plugins will install a quiet
  hook in the entry (P6.6).
- Tick order is instance-creation order; scheduling/dependencies are engine
  work (P3+), not hot-reload work.
- Multi-plugin worlds (several dylibs at once) fall out of the design
  (instances hold their own vtables) but are not yet exercised.
- Scoping stands: the Blueprint interpreter is the primary iteration loop,
  subprocess PIE (Spike D) is the crash-safe loop; hot reload is the
  compiled-preview tier and can slip without blocking the roadmap.
