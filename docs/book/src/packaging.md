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

**Cooking** turns your project into a shippable, content-addressed pack (`.inf_pack`): it resolves
the dependency closure of your levels, validates and compiles Blueprints, and bundles everything
into a deterministic archive with a GUID-sorted index. Most payloads are zstd-compressed;
streaming-class assets (virtualized meshes today, terrain tiles next) are stored uncompressed and
16-byte aligned so the runtime can read them straight out of a memory-mapped pack with no copy.
Every blob carries an xxh3-128 hash, checked the first time that entry is read. Rebuilds are
byte-identical. Cook from the CLI or from the editor:

```sh
inf cook --project MyGame --out MyGame/Build
```

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
