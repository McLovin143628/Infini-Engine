# InfiniScript — direction, and the technology ruling

**Date:** 2026-08-28. **Status:** approved direction (user mandate; wave SCRIPT0 of the
script arc). **Source:** the owner's document
`../docs/UTILIZING-CUSTOM-SCRIPTING-LANGUAGE-ALONGSIDE-RUST-TO-DEVELOP-AAA-QUALITY-GAMES-IN-A-CUSTOM-RUST-BASED-GAME-ENGINE.md`
(outside the repo), which named the engine **Infini Engine** and its scripting layer
**InfiniScript**, and asked for an architecture. **Scope:** the whole arc — SCRIPT0
(this rebrand), SCRIPT1 (the language), SCRIPT2 (the API surface + tooling),
SCRIPT3 (dogfood + ship).

This memo does two things: it adopts the source document's thesis, and it records
where the implementation departs from the document's technology picks **with the
reasons**, so neither half is relitigated later.

---

## 1. The thesis, adopted

The source document's argument, in its own terms: a AAA engine splits into a
performance core and a gameplay script layer, because

* **iteration** — a designer must change a line and see it, not wait on a compiler;
* **containment** — a broken mission script must take the mission down, not the game;
* **accessibility** — the people writing missions are not systems programmers;
* **branding** — the name is the *API surface*, not the syntax. RAGE Script is
  "the toolset that commands RAGE"; InfiniScript is the toolset that commands
  Infini Engine;
* **don't write a VM** — a studio builds the *connective tissue*, not a new compiler.

All five are adopted, unchanged, and they are the arc's brief. The document is also
right about the thing it was asked about directly: **Rust does not solve this**.
Rewriting the same architecture in Rust makes clean builds slower, not faster, and
an engine of this size will never compile in the gap between a designer's thought
and the designer's next thought.

Where this engine departs is on *which existing thing* becomes the script layer.

---

## 2. What already exists (measured, 2026-08-28)

The document's blueprint assumes a green field — `infini-script-core/` with `mlua`
and `wasmtime` in its `Cargo.toml`. This engine is not a green field. Phase 6
shipped the substrate the document describes, and Phase 14 shipped the other half:

| piece | where | what it already is |
|---|---|---|
| the IR | `crates/inf-blueprint` (`BlueprintFn`, `Stmt`, `Expr`) | a small, pure, serializable gameplay IR — events (`BeginPlay`/`Tick`/`Input`/`Collision`/`Custom`), typed data (`Bool`/`Int`/`Float`/`Str`/`Named`), exec flow |
| the interpreter | `inf_blueprint::interp` | a tree-walking evaluator **over that same IR**, running in-editor with no compile step; a failing node's damage is contained to its downstream cone |
| the compiler | `crates/inf-transpile` (`emit`, `lift`) | IR → **real Rust**, and Rust → IR back; 15 round-trip/parity test files |
| the parity gate | `inf-transpile/tests/parity.rs` | interpreted == compiled, pinned. *Preview is the shipped program.* |
| graph ↔ IR | `inf_blueprint::lower` / `raise` | **both directions**: a graph lowers to IR, and IR raises to a graph |
| the one boundary | the `Host` trait | everything external — engine calls, member variables, spawning — crosses one seam; the IR itself stays pure |
| the verb surface | `inf_blueprint::nodekit` | 96 `NodeDef::new` sites across twenty registration groups and **23** namespaces: `event.*` `flow.*` `math.*` `logic.*` `cmp.*` `var.*` `lit.*` `dispatch.*` `engine.*` `debug.*` `input.*` `audio.*` `physics2d.*` `physics3d.*` `sky.*` `water.*` `voxel.*` `destruct.*` `door.*` `item.*` `health.*` `ik.*` `anim.*` |
| the sandbox | `crates/inf-wasm-host` + `crates/inf-mod` (P14.5) | a **`wasmtime`** engine with a capability-scoped linker, a flat host ABI, and a cook path Blueprint → Rust → `cdylib` → `.wasm` |
| dylib hot-swap | `crates/inf-hotreload` | content-addressed shadow copies, never-unload, state migration |

So: the interpreter, the compiler, the parity between them, the sandbox, the
capability model and the ~100-verb API the document asks a studio to *build* are
built. What is missing is the one thing the document actually shows a designer —
**a readable text file**.

---

## 3. The ruling: no Luau, no second WASM runtime

The document's picks are **Luau via `mlua`** for the designer-facing half and
**WebAssembly via `wasmtime`** for the systems half. The second is already here.
The first is refused. Reasons of record, in order of weight:

### 3.1 A foreign VM cannot be held byte-identical, and byte-identity is the spine

`PIE == shipping` is not a slogan in this engine; it is an arm in every phase gate
since P9, and several of them compare a **byte-for-byte trace** produced by the
editor's Simulate against one produced by a real `--pie` subprocess, on three
operating systems. Everything inside the fixed step is held to that.

Luau under `mlua` brings its own garbage collector, its own numeric tower and its
own `libm`-touching math into that step. The precedent class is already on the
books and it cost real waves:

* **P14's trig law** — `f32` standard-library `sin`/`cos` are *not* bit-portable
  across platforms; committed content routes through `inf_math::psin`/`pcos`.
* **P22's extension of it** — `f64::cbrt` routes through the `libm` crate on
  `wasm32` and had to become `inf_math::pcbrt`, pinned bit-for-bit.

A script that can name Luau's `math.sin` directly is a hole straight through both
laws. Our answer is structural rather than disciplinary: **a script cannot name a
transcendental — only a verb.** There *is* a `math.sin` in the node kit and it is
not the standard library's: `inf_blueprint::math_builtins` routes it and
`math.cos` to `inf_math::portable::psin64`/`pcos64`, and `inf-blueprint`'s
`Cargo.toml` records that as the dependency's whole reason. So the determinism
guarantee is a property of the surface rather than of a review — **but the
surface is two seams, not one**, and SCRIPT1's name resolution has to know it:
pure arithmetic is dispatched by the node kit's own builtins, and everything
*external* (engine calls, member variables, spawning) crosses the `Host`. A
`.infini` identifier must resolve into one of those two and never into `std` or
`libm`.

A GC is the second half of the same problem: allocation order becomes part of the
program's observable behaviour the moment a finalizer or a table iteration order
reaches gameplay, and "deterministic across three OSes and two build profiles" is
not something an embedder can assert about someone else's collector.

### 3.2 `mlua` vendored is a C++ build in CI, and this repo has refused one before

`mlua` with the `luau` feature and `vendored` compiles Luau's C++ sources on every
runner. The precedent is explicit: **`intel_tex_2` / ISPC was refused at P4** for
exactly this — an out-of-tree native toolchain build is a cross-OS CI liability —
and BC7 was deferred rather than take it. P25 then re-paid the same lesson from a
different direction (*"one platform's bounds redden CI"*). The rule has been
consistent and the arc does not get an exemption for being interesting.

### 3.3 A fourth execution model was already rejected, on the record

`docs/memos/rust-report-crossref.md`, **Amendment 2** (pre-P14) decided this
question once already:

> Decision: **do not** add a separate embedded scripting language (Rhai/Lua/Rune).
> The blueprint tree-walking interpreter already provides no-recompile *iteration*
> over the same IR that ships as Rust, so another language would add a fourth
> execution model and fracture product principle 2 ("two ways to code, one truth").

That amendment is **not reversed by this arc — it is honoured by it.** The reason
Amendment 2 said no is that a second language means a second semantics, a second
determinism story and a second thing to keep in sync with the engine. InfiniScript
adds none of those: it adds a **parser**. It is a third *face* on the second
execution model, not a third execution model.

The one capability Amendment 2 conceded the design lacked — safe, no-compiler,
sandboxed **end-user** extensibility — is what P14.5's WASM tier serves, and it
keeps serving it under a new name.

### 3.4 The substrate is better integrated than anything we could bind

This is the affirmative half, and it is the real reason. Binding Luau would give
us an interpreter we cannot compile, cannot round-trip to a graph, and cannot
prove equal to the shipped build. We already have an interpreter that
**transpiles to native Rust** and is **gated equal to it**. A Luau script is fast
for a script; a transpiled InfiniScript function is not a script at all by the
time it ships — it is compiled Rust in the player binary, and no JIT beats that.

---

## 4. What InfiniScript is

**InfiniScript is a text front-end over the Blueprint IR.**

```
.infini text ──parse──▶ BlueprintFn IR ──▶ interpreter   (in-editor, hot-swapped on save)
                             │                              ▲
                             │                              │ parity gate
                             ├──── raise ──▶ graph ─ lower ─┘
                             │
                             └──transpile──▶ Rust ──▶ the shipped binary
```

Three consequences follow, and they are the whole design:

1. **Instant iteration with zero `rustc`.** Save a `.infini` file → the watcher
   re-lowers it → the interpreter swaps the new IR into the running Simulate. This
   is the P6 deferred item ("compile-on-save hot swap in Simulate") landing at
   last, through the interpreter rather than the dylib.

2. **The shipped program is not a script.** At cook time a script either
   transpiles to Rust into the project crate or packs its IR for the shipped
   interpreter — *both doors already exist* — and the parity gate says the two
   agree. The decision is per script class, taken with measurements, in SCRIPT3.

3. **The signature feature: graphs and text are two views of one program.** Because
   `lower` and `raise` both exist, a designer can open any Blueprint **as
   InfiniScript**, edit the text, and have it round-trip to the graph — or draw
   the graph and read the text. Unreal cannot do this; a Blueprint and a Lua
   script there are different programs in different languages. Here they are one
   `BlueprintFn` with two editors pointed at it.

**Syntax:** Luau-inspired on the surface, per the document's mockup —
`function`/`end` blocks, `Namespace:Verb(...)` calls, no sigils, readable by
someone who has never seen Rust. **Semantics: our IR exactly.** Every `.infini`
construct must lower to IR that `raise` can round-trip. The IR grows only with
pricing; the precedent is P6's own — member variables were added *via the Host
trait* with **no IR change at all**, and that is the bar.

**Determinism:** the parser is held to the same law as the runtime. A `.infini`
file's IR is a pure function of its bytes — byte-identical lowering on every host,
with a fixture that lowers the same file on two hosts and compares IR hashes.

**Refusals are values.** A parse error names its line, its column and what it
expected. It does not panic, and — per P21's law, paid for in a wave — a script
that fails at runtime takes its downstream cone and nothing else; the sim keeps
running and the editor shows the error.

### The honest bound, named now rather than discovered in SCRIPT1

`raise` is **not total**, so "two views of one program" is exactly as complete as
`raise` is. This is the whole list, read off `crates/inf-blueprint/src/raise.rs`
rather than off its summary, because SCRIPT1 plans from it.

**Of the seven-node flow palette, `raise` inverts two.** `flow.branch` becomes a
branch node; `flow.while` is recognised by `try_raise_while`, which matches the
*exact* three-statement counter-guarded expansion `lower_while` emits. The other
five do not come back:

| node | what happens on the way back |
|---|---|
| `flow.sequence` | **flattened at lowering.** The program survives exactly; the node does not. The one lossy-but-not-failing case — and the one the first draft of this memo left out |
| `flow.for` | lowers to a `Stmt::While` `try_raise_while` does not match → `UnsupportedStmt("while")` |
| `flow.do_once` | `nodestate::*` state wrapped in `Stmt::If` → `NonLinear`, or `UnsupportedExpr("pure call")` on the state read |
| `flow.flip_flop` | as above |
| `flow.gate` | as above |

**And `raise` refuses four shapes that have nothing to do with the flow
palette** — which matter *more* here than they did for the canvas, because a
canvas cannot draw them and text writes them without trying:

* `Stmt::Assign` — `UnsupportedStmt("assign")`.
* `Stmt::Snippet` — `UnsupportedStmt("snippet")`.
* a `Stmt::ExprStmt` that is not a call — `UnsupportedStmt("non-call expr stmt")`.
* a call in **value** position — `UnsupportedExpr("pure call")`. Only `engine::*`
  and `debug::*` are action paths (`is_action_path`); every other call reachable
  from an expression is out.
* and the structural one, which is the sharpest: **`RaiseError::NonLinear` — an
  `if` or a `return` that is not the last statement of its block.** `function f()
  if x then A() end B() end` is ordinary text and today it does not raise at all.

One more correction to the shape of the bound. A `Stmt::Snippet` does not sit
in a raised graph as a node-less statement: `raise_chain` returns `Err` the
moment it meets one, so **a single unraisable statement makes the whole handler
unraisable**. Lift is lossless into the *IR*; it is the graph that is all-or-
nothing per handler. `graph_open_actor` already lives with that — it opens the
handlers that raise and names the ones that do not — and that per-handler
degradation, not a per-statement one, is the precedent SCRIPT1 inherits.

Closing this gap, or bounding it and saying so in the UI, is SCRIPT1's first
named risk and not a detail. The cheap half is probably `flow.for` and
`flow.sequence` (both are shape-recognition, like `try_raise_while` already is);
the expensive half is `NonLinear`, which is a statement about what a graph *is*.

---

## 5. The compile-time truth, stated honestly

The owner asked what this does to compile times. The honest answer has three
parts and only one of them is a win today:

* **Gameplay iteration: sub-second, zero `rustc`.** Edit `.infini` → watcher →
  re-lower → interpreter hot-swap in the running Simulate. This is the number the
  arc exists to produce, and SCRIPT3 measures it rather than asserting it.
* **Engine compile times: unchanged.** Nothing in this arc makes `cargo build`
  faster. `inf-render`, `inf-vgeom`, `inf-terrain` and the rest compile exactly as
  they do now. Any claim otherwise would be false.
* **The win compounds.** As gameplay logic migrates out of Rust and into `.infini`
  over the arcs, the slow path gets hit less often — not because it got faster,
  but because fewer changes need it. That is the entire mechanism, and it is worth
  stating plainly so nobody expects the first wave to change a build clock.

---

## 6. Branding

* **InfiniScript** — the scripting ecosystem as a whole. The name of the *API
  surface*, exactly as the source document argues.
* **InfiniScript** (text) — the `.infini` front-end: what a designer writes.
  Built in SCRIPT1.
* **InfiniScript Core** — the P14.5 sandboxed WASM tier: what an engineer or a
  modder compiles. **Rebrand and document; no new runtime.** If the tier is ever
  measured to lack something, the missing thing gets priced on its own — the arc
  does not adopt a dependency in advance of a measurement.
* **The InfiniScript API Manual** — generated *from* the verb registry, not
  hand-written (SCRIPT2). The document asks for a manual; a hand-written one goes
  stale on the first verb, and the registry already carries names, categories,
  descriptions and typed pins.

---

## 7. The SCRIPT0 rename ledger, and what deliberately did not move

`Infinity Engine` → **`Infini Engine`** across everything user-facing: the title
bar, the About box, the start screen and the tour, the status bar, the window and
product names, every README, the ROADMAP headers, the mdBook, the project
templates and their scaffolded READMEs, the sample notes, the CLI and player
banners, the packager's export notes, and the crate doc comments a reader meets in
rustdoc. The repository URL moves to `Infini-Engine`; GitHub serves a permanent
redirect from a renamed repository's old path — for clones, fetches and the REST
API alike — so CI and every existing checkout keep working whichever spelling they
hold.

The sweep that produced that list was a sweep for the *phrase* — 86 of
`Infinity Engine`, 6 of `Infinity-Engine`, 3 of `InfinityEngine`. A later,
wider sweep (case-insensitive `infinity`, minus `f32::INFINITY` and the English
word) finds four more families the phrase sweep could not see. **All four
correctly did not move**, and all four are listed here because the next agent
to read this section will read it as the whole list, and "finish the job" on any
of them breaks something.

Eight things stayed, each for a stated reason.

**The crates keep their `inf-` prefix, and the repo folder keeps its name.**
Forty-odd crates, every `use` path, every `Cargo.toml`, every `include_str!`, and
every path in every memo and roadmap block would move so that a prefix which
already reads as "Infini" could read as "Infini". Zero user value; a diff nobody
can review. `inf-*` was always the right prefix and it is *more* right now.

**`com.infinityengine.app` — the Tauri bundle identifier — stays.** It is not a
label: on every platform it is the key to `app_data_dir()`, and that directory
holds the user's `Content/`, their layouts, their editor settings, their thumbnail
cache and their crash-recovery file. Renaming it would not migrate a user's work;
it would *hide* it, silently, and the editor would boot looking brand new. The
same reasoning keeps the three `InfinityEngine` filesystem roots, though not for
quite the reason the first draft gave. Enumerated:

* `inf-viewport/src/win32.rs` and `commands/diagnostics.rs` are **two writers, in
  two processes** — the viewport thread's panic handler and the editor's own
  crash hook, both falling back to `temp_dir()/InfinityEngine/crashes`. Nothing
  in the workspace *reads* a crash directory; the agreement matters because the
  human collecting a bug report needs both halves of a crash in one place, and
  moving one of the two spelling-wise would scatter them silently.
* `inf-player/src/ui.rs`'s `settings_dir()` **is** the writer-and-reader pair:
  `%APPDATA%\InfinityEngine` (or `$XDG_CONFIG_HOME/InfinityEngine`) holds
  `game-settings.toml`, which `PlayerUi::open` reads on every launch and the
  settings screen writes. Renaming it does not lose a crash dump — it resets
  every shipped player's key bindings and look sensitivity to defaults, on
  update, with no message.

So the ruling stands and its strongest case is the third root rather than the
first two.

**The generated-source marker changed, but recognition did not.**
`GENERATED_MARKER` looked like a label and is a runtime identifier:
`is_generated` is the only thing standing between the Code tab and an author's
hand-written module. Moving it with the brand would have re-classified every
already-generated file in every existing project as "not ours", and the symptom
would not have been a crash — it would have been the Code tab quietly refusing to
update files it wrote itself, in an error blaming the author. So the door now
**writes the new spelling and reads either**, with `LEGACY_GENERATED_MARKER` and
an arm that fails if the door ever forgets. *A rename may not orphan the thing it
renames.*

**"Infinity Blueprints", the themes, and the npm package name stayed.**
The theme *ids* `infinity-dark`/`infinity-light` are persisted in `localStorage`
and in `EditorSettings`, and the npm package name `infinity-engine` is a lockfile
key nobody ever sees — identifiers by the same rule as above, and the *displayed*
theme names ("Infinity Dark") are a theme brand rather than a claim about the
engine's name. The Blueprints feature name is a separate branding decision, and
this arc is precisely the wrong moment to take it halfway: SCRIPT1 makes graphs
and text two faces of one program, at which point what that one thing is called
should be decided **once**, with the text face in hand — not twice, six weeks
apart. Carried by name for SCRIPT1.

**`.infinity/` — the per-project settings directory — stays**, and it is the
same argument as the bundle identifier with the indirection removed. Six members
(`settings.toml`, `collision_layers.toml`, `sorting_layers.toml`,
`collections.toml`, `mixer.toml`, `sequences/`), spelled at six path-forming
sites — one in Ring 0 (`inf_audio::mixer::MIXER_REL_PATH`) and five in Ring 1 —
plus two asset walkers that skip it by name. It is a real directory inside
**every project that already exists** on somebody's disk, and the **cook** reads
`settings.toml` out of it. Renaming it does not migrate a project's settings; it
silently reverts them to defaults, which is the `com.infinityengine.app` failure
mode without the operating system's help.

**`#[infinity::blueprint(id = "…")]` stays**, and it is `GENERATED_MARKER`'s twin
— the same class of thing, one level down, and it did not need a legacy door only
because it did not move. `inf_transpile::emit` writes it into generated Rust in
the author's own crate, `inf_transpile::lift` reads it back
(`path.segments[0].ident == "infinity"`) to recover a function's identity when
hand-edited source is re-lifted, and `inf_packager::mods` strips it on the way to
a mod crate. Three sites that must agree about a string a *different session*
wrote into a file on disk. That is the wave's law in its purest form.

**`inf_mod::infinity_mod!` and `infinity_plugin_entry` stay** — a public macro
name and a dylib ABI symbol. `inf_hotreload::abi::ENTRY_SYMBOL` is
`b"infinity_plugin_entry\0"`, looked up by name in a plugin the host did not
compile, and `infinity_mod!` is what `samples/mods/spinner` and every documented
mod calls. Renaming either is an API break for third-party code, dressed as a
rename.

**The `infinity:` event and preference prefix stays** — six keys:
`infinity:open-file`, `infinity:asset-drop`, `infinity:rename-object`, and the
three `localStorage` keys `infinity:theme`, `infinity:tourSeen`,
`infinity:prefsMigrated`. The first three are internal wire names, the last three
are read out of a browser store a previous version of the app wrote, and
`infinity:theme` is deliberately still written (it is the pre-paint theme cache).

The Win32 window classes (`InfinityViewportClass`, `InfinityEmbedProbe`, …) were
checked and are unaffected: they are literals, not derived from the product name,
so `window_class_gate` neither moved nor needed to. No golden renders the engine's
name, so no golden moved.

The pattern in all eight is one sentence long, and it is the wave's law read
backwards: **the things that must not move are the ones written down by one
session and read back by another.** A phrase sweep cannot see them, because the
phrase is not what they are made of.

---

## 8. The wave plan

**SCRIPT1 — the language.** `inf-script` in Ring 0, **zero new external
dependencies** (the hand-rolled lexer/parser precedents are the grammar DSL and
the `.inf_sm` text face). The grammar is specced *before* it is parsed. Parse →
lower to `BlueprintFn` with source-mapped diagnostics (line/col, the naga-style
mapping). An IR → `.infini` emitter, with round-trip gates in **both** directions:
`parse(emit(f)) == f` on the IR, and `emit(parse(s))` idempotent on normalized
text. Hot reload through `inf-asset`'s notify substrate into the running
interpreter, with failure containment armed. The asset story: `.infini` is
*source* — git-diffable text — and the cook either transpiles or packs the IR,
both parity-gated, with `asset_deps` walking scripts so the cook closure sees
script-named assets (the SK1c lesson).

**SCRIPT2 — the API surface and the tooling.** The verb surface grows toward the
document's vision our way: `World.*`, `AI.*`, `Mission.*` (new, priced), `Audio.*`,
`UI.*`, `Vehicle.*`, `Weapon.*`/`Door.*`/`Item.*`. Every verb deterministic,
Host-mediated, documented. The API Manual generates from the registry. In the
editor: a CodeMirror 6 language mode for `.infini` on the P5 `extraCompartment`
seam, diagnostics wiring, open-as-text on `.inf_act`/`.inf_fn`, and Ctrl+S =
re-lower + hot-swap.

**SCRIPT3 — dogfood and ship.** Real island gameplay migrates to `.infini` — the
Phase 30 door and weapon logic, a settlement ambient script, and one
mission-class sequence at Harbour City: the document's heist mockup, made real on
our island. Migrated scripts' traces stay byte-identical to their predecessors
where semantics did not change. The interpret-vs-transpile decision is taken per
script class *with measurements*. InfiniScript Core is documented with one demo
plugin. And the arc closes honestly: the list of what a designer can build without
touching Rust, the list of what still needs Rust, and iteration timings measured
end to end — edit to running, and cold cook.

---

## 9. Laws this arc inherits

* A script cannot name a transcendental — only a verb. Pure arithmetic reaches it
  through the node kit's own builtins (which route to `inf_math::portable`) and
  everything external through the `Host`; neither seam exposes `std` or `libm`.
* `PIE == shipping` over a trace whose gameplay runs from a `.infini` script is a
  SCRIPT1 gate arm, not a SCRIPT3 aspiration.
* A gameplay refusal is a **value**, not a failure (P21) — for parse errors and
  runtime errors alike.
* No engine schema moves without a STOP-and-price. A `.inf_script` asset kind, if
  taken, is **additive** — the sidecar rule.
* A gate must aim at the thing it names, and must be built to falsify (P22/P23).
  A round-trip test that round-trips the empty program proves nothing.
