# samples/harbour-heist

**Harbour Heist** - the InfiniScript arc's dogfood mission (wave SCRIPT3):
a whole mission authored as one `.infini` file, with no Rust behind it.

- `HarbourHeist.infini` - **the mission**. Objectives, a timer, a bolted
  door, loot, stakes and an outcome, as member variables and verbs from the
  shipped node kit. Hand-authored: this file is the artifact, not a
  generated one.
- `HarbourHeist.inf_lvl` - a quayside slab, a grammar-built bank, the
  apartment block its staff live in, and one hero whose `ActorClass` names
  the script above. Everything else the mission needs - the item catalogue,
  the vault door, the bullion on the floor, the hero's own health - the
  script makes on `BeginPlay`.
- `HarbourVault.inf_pcg` / `HarbourHousing.inf_pcg` - the two buildings'
  grammar graphs. The apartment block is not scenery: `inf_ecs::society`
  pairs a HOME with a WORK to make an agent, so without it the bank has
  forty desks, nobody in them, and a mission whose `crowd.population()`
  branch could never run.
- `Alarm.inf_mesh` - **the asset the mission NAMES**. `engine.spawn` takes
  the node kit's only asset-naming string, so the cook resolves it and pulls
  this into the pack's closure; a name that resolved to nothing would BLOCK
  the build. A pack entry carries no name, so the runtime cannot resolve the
  stem back - the spawned alarm is a placeholder cube carrying the name.
  Spelling the prefab as this asset's GUID instead binds it.

The gate over it is `runtime/inf-player/tests/harbour_heist_gate.rs`, which
runs the whole mission twice on each of two routes - off a cooked
`.inf_pack` the way a shipped build boots, and off the `ScenePayload` the
editor really builds for PIE - and requires the traces byte-identical step
for step. The two routes take the two exits from the vault and reach the
two endings: loot the shelf and you are CLEAR; linger in the open, where
the staff can see you and the clock runs double, and you are CAUGHT.

The iteration claim the arc exists to make is measured on this file by
`editor/crates/inf-editor-core/tests/script_iteration.rs`: edit the mission,
and the running Simulate is running the new one.

The level, the graph and this README are generated - do not hand-edit them.
Regenerate with:

```sh
INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples
```
