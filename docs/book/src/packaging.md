# Packaging & Shipping

Infini Engine closes the make → play → ship loop deliberately early, so you can package a
double-clickable build long before the engine's advanced features are all in. Shipped games run
one execution model: compiled native code (Blueprints included, via transpiled Rust).

## Play-in-editor first

Before you ship, test with **Play-in-Editor (PIE)**. The play cluster on the main toolbar
(**Shift+Alt+P**) runs your game in a *crash-isolated subprocess*: a script panic cannot destroy
your unsaved editor state. The split-button dropdown chooses **Play (Embedded)** — the game window
docks into the viewport slot — **Play in New Window**, or in-process **Simulate** (physics and PCG
tick, no game logic). While a session runs, Pause/Resume, Step, and Stop drive it, and Eject hands
input back to you. Crucially, PIE builds the world through the *same* code path as the shipping
cook, and is proven byte-identical to it — so previewing never diverges from shipping.

## Cook a content pack

**Cooking** turns your project into a shippable, content-addressed pack (`.ipack`): it resolves
the dependency closure of your levels, validates and compiles Blueprints, and bundles everything
into a deterministic archive with a GUID-sorted index. Most payloads are zstd-compressed **whole**;
streaming-class assets — virtualized meshes, terrain, voxel volumes, world partitions and virtual
textures — are stored uncompressed *as entries* and 16-byte aligned, so the runtime can read one
page straight out of a memory-mapped pack with no copy and no whole-asset decode.

Those containers compress **per block** instead: a `.inf_terrain` tile and a `.inf_voxel` chunk each
carry their own codec in the container's directory, so paging one tile decompresses that tile and
nothing else. The saving is real — the 50 km² Vancouver Island pack is 604 631 836 B stored raw and
247 497 020 B with per-tile zstd, a 59 % reduction, and it boots *faster* because a load is
page-fault bound. Compression is lossless, so both packs simulate the same world byte for byte.

Every blob carries an xxh3-128 hash, checked the first time that entry is read. Rebuilds are
byte-identical. Cook from the CLI or from the editor:

```sh
inf cook --project MyGame --out MyGame/Build
```

Pass `--block-codec` to choose the per-block codec: `zstd` (the default, and the fastest decode of
the three that compress), `lz4` (weaker ratio, much faster decode — **the right choice for a web
build**, because zstd decodes through the pure-Rust `ruzstd` in a browser at 7.3× the cost), or
`raw` to skip the transcode entirely and ship the uncompressed containers.

In the editor, use **Build ▸ Package Project…** to open the cook dialog. The runtime reads the same
`.inf_lvl` codec the editor writes (proven byte-lockstep), so a cooked level and an editor level
are the same bytes.

## Run headless and in CI

The standalone **player** (`inf-player`) loads a pack and runs your game. It also has a headless
mode used by continuous integration and for smoke-testing:

```sh
inf-player --pack MyGame/Build --headless --run-frames 300 --assert-exit
```

This runs a fixed number of deterministic frames and asserts a clean exit, capturing a crash log if
anything fails. Infini Engine's own CI cooks the 2D platformer sample and runs it this way on
every push, on all three desktop OSes — the make → cook → play gate.

## Export a runnable build

To produce a folder you can hand to a player, use **export**:

```sh
inf export --project MyGame --out MyGame/Dist
```

This writes a renamed copy of the player, your cooked pack, and a boot config, so the exported
executable launches your content with zero arguments. Per-OS installers and code signing
(Authenticode, macOS notarization, Linux AppImage) are the packaging polish tracked in the
[release-channels design](https://github.com/McLovin143628/Infini-Engine/blob/main/docs/release-channels.md);
export gives you a working, distributable build today without faking a signed installer.
