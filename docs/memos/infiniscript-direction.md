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
| the interpreter | `inf_blueprint::interp` | a tree-walking evaluator **over that same IR**, running in-editor with no compile step; a failure aborts *that handler on that actor* and the sim ticks on (see the containment note below — the granularity is not the cone) |
| the compiler | `crates/inf-transpile` (`emit`, `lift`) | IR → **real Rust**, and Rust → IR back; 15 round-trip/parity test files |
| the parity gate | `inf-transpile/tests/{parity,flow_parity,math_parity,coyote_parity}.rs` | interpreted == compiled over **four fixture families** — see the bound below. *Preview is the shipped program, per proven family.* |
| graph ↔ IR | `inf_blueprint::lower` / `raise` | **both directions**: a graph lowers to IR, and IR raises to a graph |
| the one boundary | the `Host` trait | everything external — engine calls, member variables, spawning — crosses one seam; the IR itself stays pure |
| the verb surface | `inf_blueprint::nodekit` | **132 registered verbs** (113 `NodeDef::new` sites, three of which are the `compare`/`unary_math`/`binary_math` helpers called 22 times between them) across 23 registration groups and **26** namespaces: `event.*` `flow.*` `math.*` `logic.*` `cmp.*` `var.*` `lit.*` `dispatch.*` `engine.*` `debug.*` `input.*` `audio.*` `physics2d.*` `physics3d.*` `sky.*` `water.*` `voxel.*` `destruct.*` `door.*` `item.*` `health.*` `ik.*` `anim.*` `terrain.*` `crowd.*` `zone.*` — the last three added by wave SCRIPT2a, which also gave `engine.*` an arm recording that **neither host implements it** |
| the sandbox | `crates/inf-wasm-host` + `crates/inf-mod` (P14.5) | a **`wasmtime`** engine with a capability-scoped linker, a flat host ABI, and a cook path Blueprint → Rust → `cdylib` → `.wasm` |
| dylib hot-swap | `crates/inf-hotreload` | content-addressed shadow copies, never-unload, state migration |

So: the interpreter, the compiler, the parity between them, the sandbox, the
capability model and the 115-verb API the document asks a studio to *build* are
built. What is missing is the one thing the document actually shows a designer —
**a readable text file**.

### What the parity gate actually is, since the arc leans its whole weight on it

"Preview is the shipped program" is the arc's load-bearing claim and it is
proven **per fixture family**, not as a general property. Four files, each
running the interpreter against a **hand-written Rust mirror** of what
`generate_fn` emits, with a *string pin* tying the mirror to the generator's real
output so the two cannot silently drift:

* `parity.rs` — the rotate-on-tick handler (`vars::*` + `engine::set_rotation`);
* `flow_parity.rs` — the control-flow palette, loops and stateful nodes, over a
  persistent `nodestate` map, including the runaway guard tripping identically;
* `math_parity.rs` — the math palette, where parity holds *by construction*
  (both sides bottom out in `inf_blueprint::math_builtins`) and the sweep guards
  the coercion and dispatch around it;
* `coyote_parity.rs` — the `physics2d.*` coyote-time jump against an identical
  mock physics on both sides.

**No test in the repository compiles the transpiler's output and runs it** —
*until SCRIPT1b, which is exactly why it was routed there.* `parity.rs`'s own
module doc names the bound: it is "the CI-cheap half of the parity story (no
runtime `cargo build`)". `inf-wasm-host`'s `spinner_e2e` does compile and run
real wasm, but from the *hand-written* `samples/mods/spinner`, not from
generated code.

**SCRIPT1b closed it** (`crates/inf-script/tests/crown_parity.rs`): a `.infini`
is transpiled, the host shims are emitted beside the output, `rustc` compiles
the zero-dependency program, it runs, and its trace is required byte-identical
to the interpreter's. Everything below this paragraph about "one hand-mirrored
fixture family per construct class" therefore describes the *four families*,
which still stand and still guard what they guard; the general property they
could not give is now measured over whatever the crown gate's fixture writes.
See §8 for the two live defects that closing it found.

That is a good gate and a real one. It is also a shape SCRIPT1 must plan for:
"extending the P6 parity gate to scripts" means **one hand-mirrored fixture
family per construct class**, and the honest alternative — cook a `.infini`
script, build the generated crate, run it and diff the trace — is a new, slow,
toolchain-dependent gate that does not exist yet. Deciding which of the two
SCRIPT1 buys is a SCRIPT1 decision; pretending the first already covers
arbitrary programs is not available.

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
                                   ┌─▶ interpreter ──▶ in-editor, hot-swapped on save
                                   │         ⇕   PARITY GATE: interpreted == compiled
  .infini text ──parse──▶ IR ──────┼─▶ transpile ──▶ Rust ──▶ the shipped binary
       ▲                           │
       └──────── emit ─────────────┤
                                   └─▶ raise ──▶ graph ──▶ lower ──▶ IR
                                        ROUND TRIP: lower(raise(f)) == f,
                                        on lowering's image
```

Two different guarantees, and the first draft of this diagram put one label on
the other's arrow. **Parity** is interpreter-vs-transpiled — the `⇕` between the
IR's first two consumers. **Round-trip** is `lower ∘ raise` — the third branch,
and the thing that makes text and graph two views of one program. They fail
differently and they are gated separately.

Three consequences follow, and they are the whole design:

1. **Instant iteration with zero `rustc`.** Save a `.infini` file → the watcher
   re-lowers it → the interpreter swaps the new IR into the running Simulate. This
   is the P6 deferred item ("compile-on-save hot swap in Simulate") landing at
   last, through the interpreter rather than the dylib.

2. **The shipped program is not a script.** At cook time a script either
   transpiles to Rust into the project crate or packs its IR for the shipped
   interpreter — *both doors already exist* — and the parity gate says the two
   agree **over every construct family somebody has written a fixture for**
   (§2's bound: four families today, hand-mirrored, string-pinned to the
   generator). The decision is per script class, taken with measurements, in
   SCRIPT3.

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
expected, and it does not panic (P21's law, paid for in a wave).

**Runtime containment, at its real granularity.** The first draft of this memo
said a failing script "takes its downstream cone and nothing else". That is
`inf_graph::exec`'s property — the generic graph runner has it, with a test
called `error_is_contained_to_downstream_cone` — and the **blueprint IR
interpreter is not that runner**. `run_event` propagates a `RunError` with `?`,
so the failure aborts *that handler invocation* at the failing statement; the
containment happens one level up, where `run_on_guid` logs the error and the
per-actor loop moves to the next actor. Both hosts do this the same way
(`simulate.rs` and `runtime_sim.rs`).

The guarantee is therefore: **a broken script takes its handler, on its actor,
for that tick.** The frame completes, other actors run, the editor gets the
message. That is real containment and it is what SCRIPT1 should arm — but it is
not the cone, and a designer whose Tick script fails halfway through will find
the *rest of that handler's* effects un-run, which a "cone" model would not
predict. Narrowing it to the cone would mean running scripts through
`inf_graph::exec` instead of the IR interpreter, which is a different design and
is not what this arc proposes.

### The honest bound, named now rather than discovered in SCRIPT1

`raise` is **not total**, so "two views of one program" is exactly as complete as
`raise` is. This is the whole list, read off `crates/inf-blueprint/src/raise.rs`
rather than off its summary, because SCRIPT1 plans from it.

**Of the seven-node flow palette, `raise` inverts three** — two when this memo
was first written, and `flow.for` since wave SCRIPT1 widened it (appendix A.5 has
the reasoning and the price of the one that was *not* widened). `flow.branch`
becomes a branch node; `flow.while` and `flow.for` are recognised by
`try_raise_while` / `try_raise_for`, which match the *exact* counter-guarded
expansions the lowerer emits, through the one shared `inf_blueprint::loopshape`
matcher. The other four do not come back:

| node | what happens on the way back |
|---|---|
| `flow.sequence` | **flattened at lowering.** The program survives exactly; the node does not. The one lossy-but-not-failing case — and the one the first draft of this memo left out |
| `flow.do_once` | `nodestate::*` state wrapped in `Stmt::If` → `NonLinear`, or `UnsupportedExpr("pure call")` on the state read |
| `flow.flip_flop` | as above |
| `flow.gate` | as above |

**And `raise` refuses five shapes that have nothing to do with the flow
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

**Wave SCRIPT1 answered this**, and appendix A.5 is the answer with its gate.
The cheap half — `flow.for` — was widened, because the language grew a `for`
statement and leaving it out would have dug a hole the language itself created.
The expensive half, `NonLinear`, was priced instead of taken: closing it means
giving `flow.branch` a **join**, which the node kit does not have and the lowerer
does not emit, and re-proving `lower ∘ raise == id` over the enlarged image. That
is a wave, not a clause.

One detail above is wrong and is corrected in A.5: `flow.for` never reported
`UnsupportedStmt("while")`. It reported `UnsupportedStmt("assign")`, because a
`for` expansion satisfies the *while* matcher from its third statement. And
writing the table down as a test found a live hang in `raise` — see A.5's last
section.

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

**"Infinity Blueprints", the themes, the npm name and the crate-name fallback
stayed.** The theme *ids* `infinity-dark`/`infinity-light` are persisted in
`localStorage` and in `EditorSettings`; the npm package name `infinity-engine` is
a lockfile key nobody ever sees; and `sanitize_project`'s `"InfinityProject"`
fallback becomes a **cargo crate name** in a user's own `Cargo.toml` the moment a
project is scaffolded from an unnameable title — identifiers by the same rule as
above, and the *displayed* theme names ("Infinity Dark") are a theme brand rather
than a claim about the engine's name. The Blueprints feature name was carried to SCRIPT1 rather than
kept, because SCRIPT1 makes graphs and text two faces of one program and what
that one thing is called should be decided **once**, with the text face in hand.

**SCRIPT1 decided it: "Infinity Blueprints" → "Infini Blueprints".** The feature
is one of the engine's two authoring faces and it should carry the engine's name;
the file extension is `.infini`, the scripting layer is InfiniScript, and a
product whose flagship feature still says "Infinity" reads as a rename somebody
abandoned. Eight sites moved, all of them prose a reader meets: the README's
opening bullet and its feature table, the ROADMAP's opening paragraph and its
Phase 6 heading, the mdBook's introduction and its `blueprints-101` page, the
Spike B memo's title, `inf-blueprint`'s crate description and crate doc, and the
doc comment `inf_project::template` scaffolds into a user's own
`src/blueprints/mod.rs`. The ROADMAP's opening paragraph also gained InfiniScript
beside the graphs, since there are now three faces and it named two.

Nothing serialized moved, because nothing serialized carried the phrase — the
identifiers with "infinity" in them (`#[infinity::blueprint]`, the theme ids, the
`infinity:` event keys, `.infinity/`) are the eight families this section already
argues, and none of them is the feature's *name*.

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

Two window *titles* sit beside those classes and the wave treated them
differently, which is worth a sentence rather than a silent inconsistency: the
embed probe's `w!("Infinity Engine embed probe")` moved with the phrase, and the
viewport's `w!("Infinity Viewport")` did not — the phrase sweep could not see it.
Neither is ever drawn (the probe is deliberately not `WS_VISIBLE`; the viewport is
a `WS_CHILD` with no caption), so nothing user-facing turns on either. The
viewport's stays, deliberately: it reads as one identifier with the
`InfinityViewportClass` beside it, and splitting the pair would make that file
say two things.

**And `productName` is a label here, which is worth writing down rather than
assuming.** In general a Tauri `productName` is *not* only a label — it names the
bundled executable, the macOS `.app`, the installer and its default install
directory, so moving it can strand an existing installation beside a new one. In
this configuration it cannot: `tauri.conf.json` carries `"bundle": { "active":
false }`, so nothing is generated from it, and no Rust or TypeScript in the
workspace reads `package_info()` or the product name at all — `app_data_dir()`
resolves from the **identifier**, which did not move. The day bundling is turned
on, the install-path consequence becomes real and belongs in that decision.

The pattern in all eight is one sentence long, and it is the wave's law read
backwards: **the things that must not move are the ones written down by one
session and read back by another.** A phrase sweep cannot see them, because the
phrase is not what they are made of.

---

## 8. The wave plan

**SCRIPT1a — the language (COMPLETE 2026-08-28).** `inf-script` in Ring 0,
**zero new external dependencies** (the hand-rolled lexer/parser precedents are
the grammar DSL and the `.inf_sm` text face). The grammar is **appendix A** of
this memo, and every claim in it has an arm. Parse → `BlueprintFn` in one pass,
with diagnostics that name a line, a column and a remedy. An IR → `.infini`
emitter that is **total over the IR's shapes** — the text face is the complete
one, and the graph face is not; the emitter's six refusals (A.6) are four IR
states no producer makes plus a depth bound and one statement shape `raise`
refuses too, none of them a construct a canvas can draw. Three round-trip laws,
gated. Determinism: a digest of the fixture's IR, pinned, plus CRLF/LF/BOM
insensitivity and the crate's libm ban. The raise decision taken with pricing:
`flow.for` widened, `NonLinear` priced. **Audited the same day** — see the
SCRIPT1a audit ledger in `island-progress.md`: a language is maximum surface,
and three of its edges were crashes rather than refusals (no depth guard; an
emitter writing text its own parser rejects, from a two-node graph; an
expression in statement position).

**SCRIPT1b — hot reload, the cook, and the crown gate (COMPLETE 2026-08-29).**
All five routed clauses landed. What is now true:

* **The file door.** `inf_script::source` is the one place bytes on disk become
  a program, and the SCRIPT1a audit's routed item lands there: invalid UTF-8 is
  a diagnostic naming the **byte offset** of the first bad byte, at the line and
  column it falls on; a file over `MAX_SOURCE_BYTES` (1 MiB) is refused by
  **both numbers**; a leading byte-order mark is *repaired*, because the door's
  output is what a cook hashes. Three callers — the watcher, the cook, the PIE
  payload builder — enter through it, so a script that compiles in one compiles
  in all three by construction.
* **Where a script lives: `Content/Scripts/`**, and the ruling is IB-7's for
  IB-7's reason. A script is **content**: it takes a GUID, a sidecar, a content
  hash and a dependency closure, a level binds it through the same
  `ActorClass(Uuid)` a `.inf_act` uses, and `inf cook` opens the content root
  and nothing else. `src/` is already the *generated* Rust's home. It is a
  **convention rather than a lookup** — `AssetDb::scan` recurses, so a `.infini`
  anywhere under `Content/` is found the day it exists — so there is no manifest
  field to keep in sync. All four templates scaffold an `Example.infini`, and
  the arm compiles it rather than asserting the file exists.
* **Hot reload**, through the interpreter: watcher → door →
  `SimSession::reload_class`, drained at the **top of a fixed step** beside
  `apply_pending_tunes`, for the reason `crate::tuning` already argues. Keyed by
  **asset GUID**, because what a watcher observes is a file — **and the SCRIPT1b
  audit found that the key moved.** A payload with no sidecar is registered
  under a GUID synthesised from its **content hash**, and a `.infini` is the one
  payload in this system a human edits in place, so every save renamed it: hot
  reload queued the class under a GUID no actor was bound to and swapped
  *nothing*, and PIE, Simulate and the cook then resolved that binding to
  `None`, running the actor with no gameplay logic at all. Fixed by pinning —
  `pin_script_ids` writes the synthesised sidecar the first time a scan meets a
  script — which is what carried item 5 already said a script needs, one step
  from the feature it broke. State survives —
  `ActorInstance` sits beside the class — and variables the edit *added* are
  seeded from their defaults, or the first Tick after adding one dies at
  `vars::get`. The honest asymmetry, which a designer has to know: **changing a
  variable's default does not change a running instance.** Containment is
  tighter than §4's bound predicted: a broken edit never becomes a class, so
  nothing is queued and the previous good program keeps running with the
  diagnostics in the Output Log. The seeding rule runs **one way**, and the
  SCRIPT1b audit armed and sentenced the other two edits: a variable the edit
  *removed* **lingers** on the instance (and still wins over a later
  re-declaration's new default, because by then it is not "added"), and one
  whose *type* changed is **not migrated**, so the first handler to read it
  takes a `RunError::Type` and dies for that tick, on that actor, every tick,
  until the session is re-entered. Both are the same operation as discarding
  live state, which is what this door exists not to do.
  **Measured (corrected by the SCRIPT1b audit): the ENGINE's half — door,
  compile, swap, one fixed step — is 0.35–0.45 ms, zero `rustc`.** The plumbing
  on top is the editor's own two constants, and the wave conflated them:
  `commands/assets.rs` sets `WATCH_DEBOUNCE` to **250 ms**, and **120 ms is
  `TICK`**, the interval at which the asset thread *drains* the watcher. So edit
  → seen in the editor is `WATCH_DEBOUNCE ..= WATCH_DEBOUNCE + TICK` =
  **250–370 ms**, and the swap then lands on the next fixed step (≤ 16.7 ms at
  60 Hz). The wave's *"121 ms, of which 120 ms is the watcher's own debounce"*
  was a fact about a debounce the **arm** chose, reported as a fact about the
  editor — a number a test picks is a number about the test. The arm now reads
  both constants out of Ring 2's source, so the printed figure cannot drift from
  the shipped one.
* **The cook path, and the ship decision taken.** `AssetKind::Script` is
  additive (appended at the tail of the enum, `all()`, `FROZEN_WIRE` 24→25 and
  `kind_code` 25; no existing row moved, no schema moved). **The default is:
  the cook lowers a script and packs the IR for the shipped interpreter.**
  Reasons of record: it is what the engine already does with a `.inf_act`; the
  transpile door writes into the *user's* crate and the shipped `inf-player` is
  a prebuilt generic binary that loads packs, so transpiling changes nothing
  about what ships until somebody prices a per-project player build; and after
  the crown gate the choice costs no correctness. The transpile door stays where
  it is (the Code tab, the WASM mod cook) and SCRIPT3 takes the per-class
  decision with measurements. `i64::MIN` cooks **with the advisory** SCRIPT1a
  routed here.
* **The SK1c blocker-4 edge, open** — for `.inf_act`, `.inf_fn` and `.infini` at
  once, because all three hold the same IR and one Ring-0 walk reads it
  (`inf_blueprint::assetrefs`). What a program can name is **enumerated**:
  `STR_PORTS` classifies all twenty `Str` input ports in the node kit as an
  asset, a gameplay id, free text or a whole table, and a census arm fails if a
  verb arrives unclassified. Exactly **one** is an asset today
  (`engine.spawn`'s prefab); a name that resolves to nothing is a **blocking**
  advisory naming itself.
* **THE CROWN GATE** (`crates/inf-script/tests/crown_parity.rs`). Transpile a
  `.infini`, emit the host shims beside it, **`rustc` the zero-dependency
  program, run it**, and require the trace byte-identical to the interpreter's:
  host calls in order, every argument as a **bit pattern**, each handler's
  return, and the member-variable state after every event. 177 trace lines —
  **179** since the audit added a NaN leg through `math.min`/`math.max`, whose
  shims were a second implementation and answered `NaN` where the interpreter
  answers `2.0` — 60+ host calls, **rustc 210–234 ms** against a 60 s LOAD-class
  budget. (The
  handler-return line is `Ty::Unit` on both sides for every *event* handler, so
  the compiled side prints it as a literal; the SCRIPT1b audit made that a
  **checked** constant — the gate now asserts each driven handler is Unit and
  hands the same claim to `rustc` as `let _: () = ret;` — rather than a line one
  side cannot falsify.) `rustc` rather
  than `cargo` because there is no lock to take, no workspace to resolve and no
  manifest to write — the `inf-hotreload` fixture path needs a dedicated target
  directory and a process-private stash precisely because concurrent test
  processes race over one.
* **`PIE == shipping` over script-driven gameplay**
  (`runtime/inf-player/tests/script_gameplay_gate.rs`): an item catalogue, a
  pickup and a hand-out on a timer, cooked and run twice — off the pack the way
  `--pack` boots, and off the `ScenePayload` the editor really builds. 90 steps,
  compared per step. `build_scene_payload` needed **no new parameter**: a
  script's class arrives through the same `|guid| Option<BlueprintClass>`
  closure a `.inf_act` does. The cooked artifact's digest is pinned, so a
  lowering that depended on the host reddens one CI leg with a number.

### What the crown gate found, which nothing else in this tree can see

Both defects are invisible to every existing arm **by construction**, because
seeing them requires compiling the output, and §2's bound was that nothing did.

1. **`#[infinity::blueprint(id = "…")]` has no macro behind it.** There is no
   crate, module or macro named `infinity` anywhere in this workspace.
   `inf_transpile::emit` writes the attribute onto every generated fn and
   `inf_packager::mods` strips it with a comment saying the proc-macro *"ships
   with the engine runtime"* — it does not. So the Code tab's own output,
   `<project>/src/blueprints/<Class>_<guid8>.rs`, which the scaffolded `lib.rs`
   declares and `cargo build` compiles, **does not compile**. The comment is
   corrected, the fact is a tripwire arm, and the fix is priced rather than
   taken: either a real `infinity` proc-macro crate, or stripping at the Code
   tab's door — and the second costs `lift` its identity anchor
   (`path.segments[0].ident == "infinity"`), which is a decision with a price.
2. **`vars::get` is monomorphic in generated Rust.** The IR is untyped at the
   call and the interpreter answers with whatever `Value` is in the map; one
   Rust support function cannot return both an `f64` and a `bool`. So a member
   variable that is not a float **parses, lowers, interprets perfectly and
   cannot be transpiled** — measured by compiling one and reading `rustc`'s
   refusal. Carried; the fix is typed accessors in the emitter and a decision
   about a support module that does not exist yet.

### The bound the crown gate does not cover, named rather than discovered

**Transcendentals.** `math.sin` / `math.cos` route to
`inf_math::portable::psin64`/`pcos64`, and a zero-dependency shim cannot call
them without becoming a *second implementation* of a bit-exact polynomial —
worse than no coverage. They stay covered where they already are:
`portable_math_law.rs` proves the interpreter's routing bit for bit, and
`math_parity.rs` proves the two sides agree **by construction**. The
exact-IEEE builtins are inside the gate and are exact on both sides.

**SCRIPT2a — the call form, the API surface and the manual (COMPLETE
2026-08-29).** The wave split at the brief's own seam: 2a is the language and API
half, 2b the editor half.

* **The call form, and the pricing is its first result: it is not a wire move.**
  A handler can call its unit's own `function` declarations — A.9's loudest
  omission, retired. `Expr::Call` carries a `path: Vec<String>` and every
  registered verb is `namespace.verb`, so **a unit-local call is the same variant
  with one segment**. No new `Expr` or `Stmt` variant, no `schema_version` move,
  no `FROZEN_WIRE` row, no pack change; the arc's one schema window is not spent.
  That is §4's "the IR grows only with pricing" honoured, and the P6
  vars-via-Host precedent met a second time. The interpreter resolves it in a new
  `interp::LocalFns` decorator stacked **outside** `ActorHost` (so a callee's own
  `vars::*` still land on the actor); the transpiler needed nothing, which only
  the crown gate can say, so the gate's fixture grew in the same wave; and
  `raise` refuses it **by name** — a `NodeDef`'s ports are fixed at registration
  and a user function's signature is not, so a generic "call this function" node
  would have to be derived per class and `lower ∘ raise == id` re-proved over a
  registry that varies by document. Priced, not taken.
* **Recursion and depth are both refused statically, and for the same reason.**
  The interpreter bounds a chain at `MAX_CALL_DEPTH` (64) and answers with a
  value (P21); the transpiled Rust has no bound at all. So the two faces cannot
  be made to agree at run time, and the *language* refuses at compile time: the
  cycle, with the route it found (`ping → pong → ping`), **and** — added by the
  SCRIPT2a audit — a chain longer than the budget. Refusing the cycle alone left
  a hole the audit measured on both faces: 65 functions each calling the next
  have no cycle, so the interpreter refused at 64 while `rustc` compiled the same
  program and ran it to 65. One program, two answers. `while` is the spelling
  that survives both faces, because its bound lives in the IR (A.7).
* **The surface: 115 → 132 verbs, 23 → 26 namespaces**, every one a door Ring 0
  already had — `terrain.height_at` (dispatched by both hosts since P21.2 with no
  `NodeDef`, so nothing could call it), four `sky.*` reads, three `door.*` under
  one extracted resolution (`d3::door::nearest`), three `health.*` over a new
  `weapon::damage_entity` whose second caller is the bullet path, four `crowd.*`
  counts, and two `zone.*` queries — the mission primitive and **only** the
  primitive, with the push half (`OnPlayerEnterZone`) priced. The
  registry-versus-hosts gate now covers **every** host-dispatched namespace
  instead of the two somebody remembered, and its first finding has its own arm:
  **`engine.*` is registered and implemented by neither host** (see SCRIPT3).
* **The API Manual generates** — `docs/book/src/infiniscript-api.md`, rendered
  from the registry, **compared rather than rewritten** (`INF_BLESS_API_MANUAL=1`
  to move it), so it needs no CI job and reddens inside the ordinary battery.
  Twenty callable verbs had no description at all and now do, gated.
  `docs/book/src/infiniscript.md` is the hand-written face beside it.

**SCRIPT2b — the editor (COMPLETE 2026-08-29).** All five clauses landed, and the
sentence this memo has carried since SCRIPT1a — *"the editor surface is still the
Output Log"* — is retired.

* **The language mode is a TOKENIZER, and that is the design.** `.infini` goes
  through `languages.ts`'s one `case` and returns a `StreamLanguage` over a
  hand-rolled tokenizer mirroring appendix A.1 construct for construct, with
  **zero new npm dependencies** (`StreamLanguage`/`StreamParser` ship with the
  installed `@codemirror/language`). InfiniScript has exactly one parser and it is
  `crates/inf-script`'s; a second grammar in the frontend would be a second
  opinion about what a script *is*, and the first time the two disagreed the
  editor would be confidently wrong about a file the engine reads perfectly well.
  So it colours tokens and knows nothing about blocks, declarations, scopes or
  types. The theme bridge costs nothing: token names resolve straight against
  `@lezer/highlight`'s `tags`, so they land on the tags `cmTheme.ts` already
  paints from `--ink-*`. Its two heuristics are stated where they live — a call's
  namespace is recognised by the character AFTER the identifier, and a type name
  only after a `:` (the four are not reserved words). Two arms read `lex.rs`
  itself rather than a copied literal (`KEYWORDS`, and `SYMBOLS`/`SYMBOLS2` — 19
  symbols both sides); a third requires the shipped `Example.infini` to tokenize
  with **not one `invalid` token**.
* **The refusals cross one wire, carrying Ring 0's numbers.** `script_check(text,
  path?)` calls `inf_script::compile_bytes` — the ONE file door §8's SCRIPT1b
  entry describes — and answers with 1-based line and column, length in
  characters, severity as a word. That is what `Span`'s `Display` prints and what
  the cook prints, so the editor and the build make one claim rather than two that
  agree; the frontend converts once, where CodeMirror needs 0-based offsets. It
  takes TEXT and not a path because the dirty buffer is the program being asked
  about. **Refusals stay values across the IPC boundary** (P21) — the command has
  no `Err` path at all. Debounced at the watcher's own 250 ms, read from
  `commands/assets.rs` by the arm rather than quoted.
* **The scouted hazard is designed out rather than worked around.**
  `extraCompartment` is single-occupant and `lspBridge` reconfigures it on every
  active-tab change. InfiniScript gets a third compartment — and, more decisively,
  its contents are a pure function of the PATH, so `baseExtensions` fills it at
  `EditorState` construction and no bridge can race it. The arm drives a real
  `EditorView` through `.rs → .infini → .rs` and runs the counterfactual too.
* **Open, save, and the drawer.** A `.infini` opens through the existing
  `infinity:open-file` event from a Content-Drawer double-click; Ctrl+S writes it
  and SCRIPT1b's watcher does the rest. The designer-facing half is armed
  end to end for the first time: the path the drawer emits is the path
  `file_write` receives.
* **The graph↔text bridge's editor half fits, in one direction.**
  `script_emit_class` renders any blueprint asset as the `.infini` file it would
  be and opens it read-only. This is the arc's signature claim made operational:
  **the text face is total over the IR and the graph face is not**, so a handler
  `raise` refuses — a call form, for instance — still reads. It is read-only
  because writing back means deciding what happens to a graph-lowered class's
  synthetic local ids: `emit ∘ parse` is a fixed point on the TEXT (A.6 law 2),
  not on the `.inf_act`'s JSON, so a save would rewrite bytes the author never
  touched. **That decision is SCRIPT3's, or a wave of its own.**

Carried out of the wave, by name: no autocomplete/hover/go-to-definition over the
verb surface (the generated manual is the palette in written form until then); no
"New InfiniScript" in the drawer's Add menu (`asset_create` writes payloads and a
script is source text — a real door, priced, not taken); a `.infini` tab that has
never been displayed is never checked, because a lint source lives in a live view;
and the graph half of the bridge is still one-way.

**SCRIPT3 — dogfood, and the arc's close (COMPLETE 2026-08-29).** The wave that
says whether any of the four before it is usable. The answer is
`samples/harbour-heist/HarbourHeist.infini` — **a whole mission in one text file
with no Rust behind it**, and §10 is its ledger: what a designer can build
without Rust, what still needs Rust, the compile-time table with numbers, and
every carried item of the arc in one routed list.

* **`engine.*` is real, in both hosts** — the blocker routed here. One Ring-0
  rule each (`inf_ecs::prefab`), so the two hosts' arms diff to nothing; a
  spawned entity's identity is folded from the prefab name and the place
  (`item::authored_pickup_guid`'s ruling, met a second time) and its handle from
  that identity into **52 bits**, so it survives `math.to_float` into a `float`
  member variable — above `2^53` it would round on the way in and name a
  different entity on the way out. Both consequences this entry predicted are
  paid: the tripwire arm and the registry gate's `engine` exception are retired,
  and `build_scene_payload` walks a class's `asset_refs`. **Only a GUID-spelled
  prefab binds at run time**, and that is the verb's contract rather than a
  shortcut: a `PackEntry` carries no name, so a shipped player has nothing to
  resolve a file stem against.
* **The mission**, gated `PIE == shipping` **byte-identical over two routes**
  with two exits and two endings (clean: six bars, out on the loot, clear;
  interrupted: two bars, out on the clock, **caught**). Three engine facts cost
  the wave real time and are in the ROADMAP block: an archetype has a lot size
  it plans properly at, `society` pairs a HOME with a work to make an agent, and
  the first PIE payload resolved one grammar graph of two — which diverged the
  trace **at step 1** and is exactly what that gate exists to catch.
* **The iteration claim, measured on that mission**: **1.08–1.33 ms** of engine
  time to turn a saved edit into new behaviour mid-heist, with the run's state
  intact across the save. §10.3 has the whole table.
* **The `OnPlayerEnterZone` event is re-priced, not taken.** The mechanical
  items were right and were not the hard part: **the hard part is where a zone
  lives**. Routed to SCRIPT4 with three homes priced.
* **Not done, and named**: the Phase 30 door/weapon logic is not migrated, there
  is no settlement ambient script, and InfiniScript Core still has no demo
  plugin. All three are SCRIPT4's, in §10.5's list.

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

---

## 10. The arc's close (wave SCRIPT3, 2026-08-29)

The arc set out to answer one question the owner asked: **can a designer build a
game in this engine without waiting for a compiler?** Four waves built the
language, the cook, the hot-reload path, the editor and the verb surface —
**132 registered nodes, of which 94 are callable verbs**; the other 38 are the
29 written-as-syntax refusals and the 9 `event.*` headers, which is the
accounting the SCRIPT2a audit closed and the shorthand "132 verbs" quietly
re-opens. This section is what the fifth wave measured, on a real mission rather
than on a fixture.

The artifact is `samples/harbour-heist/HarbourHeist.infini` — 5 619 bytes, two
handlers, six functions, committed island content with its own level, its own
`PIE == shipping` gate and its own measured edit loop. It has objectives, a
timer, a bolted door, loot, a crowd that reacts, stakes and two endings. **There
is no Rust behind it.**

### 10.1 What a designer can build without touching Rust

Read off the mission's own verbs rather than off the registry, because a list
taken from the registry is a list of what exists and this is a list of what got
used in anger.

| the mission needed | the verbs it used | and what that generalises to |
|---|---|---|
| an objective — *am I in the vault, am I at the boat* | `zone.contains` | any trigger region, polled on `Tick` |
| a **state machine** | member variables + `if`/`elseif` + the SCRIPT2a **call form** | phases, objectives, fail states, anything a designer would draw as a flowchart |
| a clock | `Tick`'s own `dt` and a `float` variable | timers, cooldowns, grace periods — deterministic, because `dt` is the fixed step |
| a door that bolts behind you | `door.spawn`, `door.lock`, `door.use` | doors, gates and hatches — every leaf in the level, hung by a script or by the building grammar. *(Not "the whole `Interactable` surface", which is what this cell first said: nothing in the kit authors a **generic** `inf_ecs::interact::Interactable`. The only two verbs that make one are `door.spawn` and `item.spawn_pickup`, and a switch or a terminal still needs a component.)* |
| loot | `item.define`, `item.spawn_pickup`, `item.give`, `item.count` | catalogues, pickups, inventories, ammunition |
| stakes | `health.set`, `health.damage`, `health.fraction` | damage, healing, downing, any resource with a maximum |
| a world that answers back | `crowd.population` | population pressure, "is anyone watching", "is the town alive" |
| putting things in the world | `engine.spawn`, `engine.destroy`, `engine.set_rotation` | markers, effects, props, anything a mission needs to exist for a while — **at the acting entity's own place**, because the verb takes no point (see §10.4's second bound) |
| telling the player | `debug.print` | (see the honest list below — this is the surface's weakest point) |

And beyond the mission, from the same registry and with the same standing:
`sky.*` (time of day, weather, fog), `terrain.height_at`, `water.*`, `voxel.*`
carving, `destruct.*`, `anim.*` and `ik.*`, `audio.*`, `physics2d.*` /
`physics3d.*`, `input.*`, and `dispatch.*` for actor-to-actor events. The whole
of it is in `docs/book/src/infiniscript-api.md`, generated from the registry and
drift-checked in the ordinary battery.

**And the part that is not a verb list**: the mission is *hot-swappable while it
runs*. That is what "without touching Rust" is worth — see §10.3.

### 10.2 What still needs Rust

Four things, and the first is the one a designer meets first.

1. **A user interface.** There is no `ui.*` namespace. The mission tells the
   player things with `debug.print`, which goes to the Output Log. Objectives on
   screen, a timer, a damage flash, a "MISSION FAILED" card — all of it is
   engine work today. `inf-ui` exists; a `ui.*` kit over it is the single
   highest-value verb family this arc did not build.
2. **Locomotion, and anything that owns a frame.** *(Corrected by the SCRIPT3
   audit. This entry first read "a script reacts on `Tick`; it cannot drive a
   character", and the registry says otherwise.)* A script **can** move a body:
   `input.is_down` on `Tick` into `physics3d.move_and_slide` is a working
   controller, and `simulate_character3d.rs` drives one over rapier3d through
   exactly those two verbs. What has no verbs at all is **`CharacterMovement`
   and the P29 catalogue** — gaits, locomotion states, root motion, the fourteen
   modes — and nothing in the surface owns a camera or a frame. The Harbour
   Heist gate moves its hero from outside the mission for a *gate's* reason
   rather than a language one, and says so in its own header: a
   player-controlled mover would fold gravity, ground snap and a camera into
   every step of the trace, and this is a mission gate.
3. **New engine capability.** A verb is a door onto something Ring 0 already
   does. Where the door does not exist — a trigger volume that pushes an event,
   a nav query, a spatial index over the crowd — the door is Rust, and the whole
   arc's rule is that a verb is added *when* the door exists (SCRIPT2a added
   seventeen and every one had a `pub fn` behind it already).
4. **Data structures.** The IR's value set is `Float`/`Int`/`Bool`/`Str`. No
   arrays, no tables, no structs (A.9). A mission that wants "the three nearest
   guards" cannot hold them.

### 10.3 The compile-time table — the owner's question, answered with numbers

Every figure is measured on this machine, and **nothing here is asserted in a
test** (the house rule about wall clocks). The three *provenances* are separated
rather than glossed, because this paragraph first said "everything here is
printed by [an arm]" and three of the six rows are not:

* **printed by an arm** — the two script rows and the cook row, by
  `script_iteration.rs`, `script_hot_reload.rs` and `harbour_heist_gate.rs`;
* **timed by hand at a shell** — the two warm `cargo build` rows. No arm runs
  them and no arm could, because an arm that did would be a wall-clock
  assertion;
* **history, not a fresh measurement** — the cold row, quoted from CLAUDE.md's
  machine notes, and it is the whole **battery** and the whole **clippy** pass
  rather than a rebuild of one crate.

| the road | what it costs | measured by |
|---|---|---|
| **edit a `.infini`, running Simulate** — door + parse + lower, swap, one fixed step, on the 5 619-byte mission | **1.08–1.33 ms**, zero `rustc` | `script_iteration.rs` |
| …plus the editor's plumbing before it: the watcher's debounce, plus up to one drain tick | **250–370 ms** | `script_hot_reload.rs`, read out of Ring 2's own source |
| **cook the whole mission project** — two grammar graphs, a level, a mesh and the script | **156 ms** | `harbour_heist_gate.rs` |
| **change one Ring-0 gameplay rule** and rebuild the crate Simulate lives in, warm incremental | **7 s** | `cargo build -p inf-editor-core` |
| …and the shipped player with it | **12 s** | `cargo build -p inf-player` |
| …from cold | **~50 min** for the battery, **~35 min** for clippy | CLAUDE.md's machine notes — **history**, not measured this wave |

The engine's half of the script road is about **5 800×** cheaper than the warm
Rust road: 7 s ÷ ~1.2 ms. What a designer *feels* is the plumbing on top, and
**the divisor is worth stating rather than rounding** (the SCRIPT3 audit's
correction — the two halves of this sentence used to disagree by 1.5×):
**7 s ÷ 370 ms is 19×**, the slow end, a save whose debounce has just missed a
drain; **7 s ÷ 250 ms is 28×**, the fast end. So a quarter-second to a third of
a second against seven seconds — and the seven seconds is the *warm* number,
which is the one a good day gives you.

The three statements §5 made are all still true and now have arms under them:
gameplay iteration is sub-second with zero `rustc`; **engine compile times are
unchanged**, because nothing in this arc makes `cargo build` faster; and the win
compounds as gameplay migrates out of Rust, because the slow path gets hit less
often rather than getting quicker.

### 10.4 The bounds that showed up in a real program

SCRIPT1b's carried item 2 said *"a member variable that is not a float parses,
lowers, interprets perfectly and cannot be transpiled"*, measured on a `bool`.
Writing a mission sharpened it into something much bigger:

**No script that names its own entity can be transpiled.** The host seeds
`entity` with an `Int`, every gameplay verb that needs a subject takes it, and
`vars::get` is monomorphic in generated Rust. The Harbour Heist mission has five
non-float member variables and one of them is `entity`, which no author chose.

This costs the arc nothing *today*, because the ship decision SCRIPT1b took is
that the cook packs IR for the shipped interpreter (§8) — the mission ships, the
gate proves it ships identically to PIE. What it costs is the **Code tab**: a
designer cannot open a real gameplay script as Rust. Carried, routed, and now
with a real program's shape attached to it rather than a two-line fixture's.

**And a second bound, found by the SCRIPT3 audit reading the ruling above it.**
A spawn's identity is `(prefab, place)`, so two spawns of one prefab at one
point are one entity — and **`engine.spawn` takes no point.** It spawns at the
acting entity's own position, which means the remedy the first draft of
`inf_ecs::prefab`'s header offered ("an author who wants two puts them in two
places") is not reachable from inside a script: an author needs two spawner
actors, or a spawner that has moved between the two calls, or a second name.
Where the *point* is the thing being authored, `item.spawn_pickup` is the verb
that takes one. Closing it properly is a second input on the node — a kit
change, priced here and not taken — and the honest sentence is now in the verb's
own description, so the generated manual carries it.

### 10.5 The arc's carried items, consolidated

Every **wave** list the arc carried, in one place, with a route. Nothing here is
new in SCRIPT3; what is new is that they are in one list instead of five.

**The scope, stated rather than implied** — the SCRIPT3 audit's correction, and
it is the ordinary shape of a consolidation claim. This paragraph first read
*"everything the five waves carried"*, and what the list actually consolidates
is the five **wave** ledgers' own "Carried, by name" sections. Two things sat
outside it and are now inside or pointed at:

* three items from SCRIPT1a's wave list that SCRIPT2a's item 9 explicitly
  re-affirmed as standing were dropped on the way in. They are **16–18** below;
* the arc's four **adversarial audits** each carried a list of their own, and
  those are **not** folded in here — SCRIPT1a's six LOWs, SCRIPT1b's three,
  SCRIPT2a's four and SCRIPT2b's eleven, **twenty-four items**, standing where a
  reader of that wave meets them in `docs/memos/island-progress.md`. They are
  message-honesty, editor-UX and fixture-shape items rather than surface
  decisions; a wave that wants them reads those four sections and not this one.

**Routed to SCRIPT4 (a wave's worth of work each):**

1. **`ui.*`** — §10.2's first entry, and the surface's biggest hole.
2. **The `OnPlayerEnterZone` event.** Re-priced in SCRIPT3 from the other side:
   the five mechanical items are right and are not the hard part; **the hard
   part is where a zone lives**, because a pushed event needs the fixed step to
   know the boxes before any handler runs, and today a box is six floats inside
   a script's own expression. Three homes are priced in `nodekit`'s zone doc — a
   `TriggerVolume` component (right long-run, a scene schema move), a
   `zone.watch` resource the script declares (no schema move, the `item.define`
   precedent, the recommended first cut), or keep polling (one fixed step of
   latency, and a mission that reads better as an explicit state machine).
3. **`RaiseError::NonLinear`, and the graph face's other refusals** (A.5). Closing
   it means giving `flow.branch` a join and re-proving `lower ∘ raise == id` over
   the enlarged image.
4. **The graph↔text bridge is one-way.** A Blueprint reads as InfiniScript;
   writing that text back into the graph needs a decision about a graph-lowered
   class's synthetic local ids (SCRIPT2b).
5. **`raise` has no call-a-function node** and cannot have a generic one: a
   `NodeDef`'s ports are fixed at registration and a user function's signature is
   not (SCRIPT2a).
6. **Migration.** The Phase 30 door/weapon logic is still a `.inf_act`, there is
   no settlement ambient script, and re-authoring committed content whose gate
   pins its bytes is a wave rather than a clause.
7. **InfiniScript Core** — the P14.5 WASM tier is rebranded and documented, and
   has no demo plugin of its own.

**Maintenance-class. Five of these have a live arm that fires the day they
change — 8, 9, 10, 12 and 13 — and the rest are honest sentences, marked *(no
arm)*. That split is the SCRIPT3 audit's correction too: this heading read "each
with a live arm", and an item with no arm under a heading that promises one is
exactly the blind spot with a name on it that an exception list is:**

8. **`#[infinity::blueprint]` has no macro behind it**, so the Code tab's output
   does not compile (`the_generated_marker_has_no_macro_behind_it`). Priced:
   either a real proc-macro crate, or stripping at `blueprint_source`'s door —
   and the second costs `lift` its identity anchor.
9. **`vars::get` is monomorphic in generated Rust** — see §10.4.
10. **A `string` parameter on a unit-local function does not transpile**
    (SCRIPT2a): `Ty::Str` renders as `String` and `Lit::Str` as a `&str`, and the
    two do not meet at a generated call.
11. *(no arm)* **Transcendentals are outside the crown gate**, by a stated
    argument rather than an oversight: a zero-dependency shim cannot call
    `psin64` without becoming a second implementation of a bit-exact polynomial.
    SCRIPT1b's audit already classified this one as an honest sentence.
12. **`crowd.*` counts and does not name an agent**; **`zone.count` counts
    everything with a collider**, the ground included.
13. **A prefab name does not bind at run time.** The cook resolves a file stem
    because it has the asset database; a `PackEntry` carries no name, so the
    shipped player cannot, and a spawn names a placeholder cube unless the
    prefab is spelled as a GUID. Closing it is a pack-format move.
14. *(no arm)* **Editor surface**: no autocomplete/hover/go-to-definition over
    the verbs; no "New InfiniScript" in the drawer's Add menu; a `.infini` tab
    never displayed is never checked; `--ink-danger` is undefined at 18 sites
    across 8 files (SCRIPT2b). Nothing in the frontend suite pins any of the
    four.
15. *(no arm for the list as a whole)* **A.9's v1 language omissions**: no
    tables/arrays/structs, no closures, no `repeat`/`break`/`continue`, no
    string concatenation, no multiple returns. Each is a pricing question about
    the IR, and the question to ask first is what the shapes it already has
    cannot say — which is how the call form turned out to need no IR change at
    all. (Some of the *refusals* are armed; the absences are not, because an
    absence has nothing to assert against.)
16. *(no arm)* **A `math.neg` node wired to a `lit.float` lowers to IR the
    transpiler refuses** (SCRIPT1a's carried item 4, re-affirmed by SCRIPT2a's
    item 9 and dropped on the way into this list). The *language* no longer
    writes `Unary(Neg, Lit)` — the parser folds it into a negative literal —
    but the **graph** lowerer still can, so the Code tab would fail to generate
    for such a graph. A.5's second bullet has the reasoning; the fix is a fold
    at lowering, or teaching `emit` the canonicalisation and re-proving
    `generate → lift`.
17. *(no arm)* **`math.pow` is `f64::powf` and is not bit-portable** (SCRIPT1a's
    item 6, likewise re-affirmed and likewise dropped). It is the one hole in a
    math palette the P14 determinism law otherwise closes, it is inherited from
    the node kit rather than opened by this arc, and it is **said in the verb's
    own description** so the generated manual carries the warning. A committed
    value must not depend on it. The remedy is a portable polynomial in
    `inf_math`, on `psin64`'s pattern.
18. *(no arm)* **The source map is handler-level** (SCRIPT1a's item 7, the
    third). `Unit::spans` records where each handler begins, which is the
    granularity containment actually has (A.7); a statement-level map is what a
    panel would want the day one asks for it.
19. *(armed — `harbour_heist_gate`)* **The mission's gate depends on the crowd's
    determinism as well as the script's.** `crowd.population()` is a mission
    mechanic, so 28 agents grown from two grammar-built buildings are inside the
    compared trace. That is a property this repository already gates
    (`island_gate`, `crowd_sweep`), and it is named because a crowd inside a
    *script* gate is new: if the crowd ever stops being deterministic, this gate
    goes red for a reason that is not about scripting. The coupling is stated in
    the gate's own module header, which is where a crowd wave will meet it.
20. *(no arm; the cook prints the second half as an advisory)* **The alarm mesh
    is packed and nothing draws it**, and it takes **two** things to change
    rather than one. It is in the pack because the mission names it and the
    closure works; it is not on the spawned entity because a stem does not bind
    (item 13). And even spelled as a GUID it would still draw a placeholder,
    because the cook says so in the mission's own report: *"mesh … (Alarm) has 12
    triangles, below the virtualized-geometry threshold of 2048, so no
    `.inf_vmesh` was derived — the shipped build renders it as a PLACEHOLDER
    CUBE"*, which is P23's own carried finding met on this content. So the
    SCRIPT3 ledger's *"the day item 1 closes, the same content starts drawing"*
    is one closure short: the pack needs a name index **and** the sample needs a
    denser mesh (or a lowered `[vgeom] min_triangles`). Corrected by the SCRIPT3
    audit, off the gate's own printed output.

### 10.6 The verdict, in one paragraph

A designer can build a **mission** in this engine without touching Rust: places,
objectives, timers, doors, items, damage, a reacting crowd, things spawned and
destroyed, and a state machine over all of it — edited live, in a running world,
in about a millisecond of engine time and a quarter-second of editor plumbing,
against seven seconds for the warm Rust road and fifty minutes for the cold one.
What they cannot build without Rust is a **user interface**, **locomotion** (a
body they can move with `physics3d.move_and_slide`; a `CharacterMovement` gait
they cannot), a new engine capability, or a data structure. The first of those
is one wave's work over a substrate that already exists, and it is the thing to
build next.

---

# Appendix A — the InfiniScript language, v1

**Status:** implemented in `crates/inf-script` (wave SCRIPT1a, 2026-08-28).
Every claim in this appendix has an arm, and the arm is named beside the claim.

The rule the whole appendix hangs from: **a `.infini` construct is a spelling of
an IR construct, never more.** There is no separate syntax tree — the parser's
output type *is* `inf_blueprint::BlueprintFn`, and `Stmt`/`Expr` are the AST.
Anything sitting between the two could acquire semantics of its own, which is
exactly what Amendment 2 refused.

## A.1 Lexical

* **Encoding** is UTF-8. **A leading byte-order mark is dropped, and line
  endings are normalised** (`\r\n`, and a lone `\r`, both become `\n`) before
  anything else, so a Windows checkout and a Linux one lower a file to the same
  IR — `crates/inf-script/tests/determinism.rs`'s
  `a_windows_checkout_and_a_unix_one_lower_identically` and
  `a_file_saved_with_a_byte_order_mark_lowers_identically`. Only the *leading*
  mark: U+FEFF elsewhere is a zero-width no-break space, which inside a string
  is content and outside one is an unexpected character with a place. (The BOM
  was the SCRIPT1a audit's find — unstripped it was `unexpected character` naming
  a character invisible in its own message, on a file Notepad had just saved.)
* **Comments** run from `--` to end of line. There are no block comments, and
  **a comment does not survive a round trip.** The parser's output type *is*
  `BlueprintFn`, which is the thesis and which also means there is nowhere for
  trivia to live — so `emit(parse(src)) == src` (A.6, law 1's corollary) holds
  over the committed corpus *because that corpus carries no comments*, and
  "open a script, save it, no diff" is true of a file that has none. Pinned by
  `roundtrip.rs`'s `a_comment_does_not_survive_the_round_trip`, which costs
  nothing until SCRIPT2's graph→text door writes a `.infini` back.
* **Nesting is bounded** at `parse::MAX_NESTING` = **128** levels per
  declaration — blocks, parentheses, unary operators and the steps of an
  operator chain share one budget, because they share one stack. Past it is a
  refusal naming the limit and the remedy. This is P19's parser-depth-guard law,
  and the SCRIPT1a audit measured what its absence cost: on a **1 MiB** stack (a
  Windows main thread) the parser died between 512 and 768 nested parentheses
  with `STATUS_STACK_OVERFLOW` — not a refusal, a *crash*, in a library the
  editor calls on every keystroke. A chain spends the budget too (`1 + 1 + … + 1`
  needs no parser frames and still builds an `n`-deep tree, which killed the
  *emitter* at ten thousand terms), because the bound is on the tree that every
  downstream consumer walks. The deepest construction in the round-trip corpus
  is 4. Arms: `tests/hostile.rs`.
* **Reserved words** (20): `actor and do else elseif end exposed false for
  function if local not on or return rust then true while`. `var` is
  deliberately *not* reserved — a declaration reads `var speed: float = 0` and
  the escape hatch reads `var.get("hit count")`, and the second is a call whose
  first segment has to be an identifier. It is contextual: at the top level a
  leading `var` opens a declaration, everywhere else it is a name. `local var`
  and `local nodestate` are refused so the two readings can never collide.
* **Identifiers** are `[A-Za-z_][A-Za-z0-9_]*`, plus alphabetic Unicode.
* **Numbers**: a literal containing `.` or an exponent is a `float`, otherwise an
  `int`. The token keeps its source text unparsed, so `-9223372036854775808`
  (whose digits alone overflow an `i64`) has a literal form.
* **Strings** are `"…"` on one line, with `\n \r \t \0 \\ \" \u{…}`. A string may
  not span lines; write `\n`, or use a long bracket.
* **Long brackets** `[[ … ]]`, `[=[ … ]=]`, `[==[ … ]==]` — Lua's form at any
  level, carrying `rust` blocks verbatim. One newline immediately after the
  opener is not content. The emitter picks the shortest level the content does
  not contain, which is what makes the form total over opaque Rust: `vec![[1,
  2][0]]` really does contain `]]`.

## A.2 Grammar

```
unit        := ('actor' STRING)? decl*
decl        := 'var' IDENT ':' TYPE '=' literal 'exposed'?
             | 'on' event params block 'end'
             | 'function' IDENT params ('->' TYPE)? block 'end'
event       := IDENT | 'input' STRING | 'custom' STRING
params      := '(' (IDENT (':' TYPE)? (',' …)*)? ')'
TYPE        := 'float' | 'int' | 'bool' | 'string'

block       := stmt*
stmt        := 'local' IDENT (':' TYPE)? '=' expr
             | IDENT '=' expr
             | 'if' expr 'then' block ('elseif' expr 'then' block)* ('else' block)? 'end'
             | 'while' expr 'do' block 'end'
             | 'for' IDENT '=' expr ',' expr 'do' block 'end'
             | 'return' expr?                    -- the value on the SAME line
             | call
             | 'rust' LONGBRACKET
             | ';'

expr        := or
or          := and ('or' and)*
and         := cmp ('and' cmp)*
cmp         := add (('==' | '~=' | '<' | '<=' | '>' | '>=') add)?     -- no chaining
add         := mul (('+' | '-') mul)*
mul         := unary (('*' | '/' | '%') unary)*
unary       := ('-' | 'not') unary | primary
primary     := NUMBER | STRING | 'true' | 'false' | '(' expr ')' | call | IDENT
call        := IDENT ('.' IDENT){0,2} '(' (expr (',' expr)*)? ')'
```

**Zero dots is the call form** (wave SCRIPT2a): a bare `name(args)` naming one of
this unit's own `function` declarations. It resolves against the headers the
parser prescans, so a handler written above its declarations still finds them;
arity is **exact** (a verb's trailing inputs have lowerer defaults, a function's
parameters have none); and value position requires a `->`, because a function
with no return type has no value to be. A bare name that matches no declaration
is still the registry's refusal, now saying how to declare one.

A returned value must be on the `return`'s own line. Without the rule, `return`
followed by a call on the next line reads the call as the returned value. Lua
avoids the ambiguity by requiring `return` to end its block; this IR has
`Stmt::Return` mid-body — it is `raise` that refuses one, not the interpreter —
so the line is the cheaper rule, and it matches what a reader sees.

## A.3 Names

A bare identifier resolves innermost-first:

1. a `local` in an enclosing block → `Expr::Local`;
2. a handler or function parameter → `Expr::Param`;
3. **anything else is a member variable** → `vars::get("name")`.

Rule 3 is total rather than a lookup, which is what lets one handler be parsed on
its own — the graph↔text bridge opens a single function, with no declarations in
sight — and still mean what it means inside its class. When the whole unit *is*
in view and declares no such `var`, that is a **warning**, not an error: the
runtime already refuses an unknown variable by name, which is P21's law.

**Shadowing is refused** — a deliberate divergence from Lua and from Rust. Two
live bindings of one name print as one name, and a re-parse would bind every
reader to the later one: a silent change of program across a round trip. Renaming
costs a designer one word.

`var.get("…")` / `var.set("…", v)` are the explicit forms, for a member variable
whose name is not an identifier. `nodestate.get_or(k, d)` / `nodestate.set(k, v)`
are the lowerer's own state cells, which appear when a graph using
`flow.do_once` / `flip_flop` / `gate` is opened as text. **The spelling rule:
text uses the node `type_id` where a node exists (`var.get`, `dispatch.call`) and
the host path where none does (`nodestate.get_or`).**

**Mutability is derived.** There is no `local mut`; a local is `mutable` in the
IR exactly when something assigns to it.

## A.4 Calls, and the two seams

A call names a namespace and a verb — `debug.print("hi")` — and the spelling is
the node kit's `type_id`. Resolution happens **at parse time** against the
registry (`crates/inf-script/src/verbs.rs`), which is what makes the determinism
guarantee structural: there is no path from `.infini` text to `std` or `libm`.
The surface it resolves into is **two seams, not one**:

* `math.*` (except the operators) is **hostless** — `dispatch_math` evaluates it
  straight out of `inf_blueprint::math_builtins`, the same functions the
  transpiled Rust calls, which is why the math palette passes parity by
  construction. `math.sin`/`math.cos` route to `inf_math::portable`.
  `tests/portable_math_law.rs`'s
  `a_script_that_writes_math_sin_gets_the_portable_one` proves it by running a
  script over a `Host` that panics if it is reached, then comparing bits.
* everything external crosses the one `Host` trait.

A multi-result query names its result as a third segment
(`physics2d.raycast.hit(…)`) — the lowerer's own rule that one wire carries one
scalar. Arity and required arguments are checked at parse time.

And **a call with no namespace at all is the third thing a call can be**: the
call form (A.2), a bare name resolving to one of this unit's own `function`
declarations, on the same `Host` seam. It is checked *before* the registry, so a
declaration in the file wins, and it can never collide with the two synthesized
namespaces (`vars`, `nodestate`) because those are two segments and this is one.

**Six families of registered node are deliberately not callable**, each refused
with the syntax that replaces it: the arithmetic/comparison/logic operators (they
are `+`, `<`, `and` here), `logic.not` and `math.neg`, the `flow.*` palette
(control flow is syntax), `event.*` (an event is a handler's header), `lit.*`
(write the value), and `var.get`/`var.set` (write the name).

The surface is **132 verbs across 26 namespaces**, pinned by `inf_script::verbs`'
own census arm.

That arm's message names this sentence, and the SCRIPT2a audit found the promise
was worth less than it read: the wave moved the census to 132/26, the arm's
numbers moved with it, and **this line stayed at 115/23 for a whole wave**. A
census arm pins the *number a test compares*; it cannot pin prose in a file it
never opens. So "cannot go stale silently" is downgraded to what is true — the
arm fails the day the registry moves, and updating the two places that quote it
in prose (here and §2's table) is the wave's job. The third quotation — the
generated manual's own header line — really is checked, by
`the_manual_covers_the_surface_it_claims_to`, which is the difference between a
number a test owns and a number a memo repeats.

## A.5 The honest-subset table — what raises, and what only exists as text

`raise` (IR → *graph*) is not total, so "graphs and text are two views of one
program" is exactly as complete as `raise` is. **The text view is the complete
one**: the emitter refuses nothing the IR can hold. This table is executed by
`crates/inf-script/tests/raise_coverage.rs`; the rows below are its rows.

| construct | text | graph |
|---|---|---|
| a chain of actions; member variable reads and writes | ✔ | ✔ |
| the operators, unary minus, `not` | ✔ | ✔ |
| a pure `math.*` builtin in value position | ✔ | ✔ |
| `if … then … [elseif …] [else …] end` **as its block's last statement** | ✔ | ✔ |
| `while … do … end`, with or without statements after it | ✔ | ✔ `flow.while` |
| `for i = a, b do … end` | ✔ | ✔ `flow.for` — **widened in SCRIPT1** |
| `return` as its block's last statement | ✔ | ✔ |
| an action bound to a local (`local e = engine.spawn(…)`) | ✔ | ✔ *(and see the hang, below)* |
| **an `if` or `return` that is NOT its block's last statement** | ✔ | ✘ `NonLinear` |
| **assigning to a local** (`local x = 0` then `x = x + 1`) | ✔ | ✘ `UnsupportedStmt("assign")` |
| **a `rust [[…]]` block** | ✔ | ✘ `UnsupportedStmt("snippet")` |
| **a non-action call in value position** (`physics2d.is_grounded(e)`) | ✔ | ✘ `UnsupportedExpr("pure call")` |
| **`nodestate.get_or` / `nodestate.set`** (a `do_once`/`flip_flop`/`gate` graph, read as text) | ✔ | ✘ `UnsupportedExpr("pure call")` |
| **the call form, in statement position** (`tap()`) — SCRIPT2a | ✔ | ✘ `LocalFunctionCall("tap")` |
| **the call form, in value position** (`speed = two()`) — SCRIPT2a | ✔ | ✘ `LocalFunctionCall("two")` |
| `flow.sequence` | — | flattened at lowering: the program survives, the node does not |

The two SCRIPT2a rows are a **named** refusal rather than a folded one, and that
is the point: before the call form existed, a one-segment call in statement
position went to `raise_action`, which added a graph node whose `type_id` no
registry knows — a silently malformed graph rather than a verdict — and in value
position it was folded into the generic `pure call`, whose remedy does not apply.
Naming it also makes the editor's degradation honest: `graph_open_actor` reports
every handler with a `raisable` flag and a `reason`, so a call-form handler
appears in the switcher, greyed, saying what it is.

Two consequences a designer should be told rather than discover.

**One unraisable statement makes the whole handler unraisable.** `raise_chain`
returns `Err` at the first one, so a handler with fifteen good statements and one
assignment opens as text and not as a graph. That is the per-handler degradation
`graph_open_actor` already lives with, and it is the granularity SCRIPT2's UI has
to speak in (`one_unraisable_statement_takes_the_whole_handler`).

**Member variables are the graph-friendly way to hold state.** A local assignment
has no graph form, because a graph value is a wire and a wire is not re-assigned;
`speed = speed + 1` on a member variable raises perfectly.

### What SCRIPT1 widened, and what it priced instead

`flow.for` was widened, because the language grew a `for` statement and leaving
it out would have dug a hole the language itself created. It cost one matcher in
`inf_blueprint::loopshape`, shared with the lowerer so the two cannot drift.

`RaiseError::NonLinear` was **not**, and the price is stated rather than implied.
Closing it means giving `flow.branch` a **join** — a merge point where its two
exec paths come back together — which the node kit does not have, the lowerer
does not emit, and `lower ∘ raise == id` would then have to hold across. New port
semantics on a frozen node, a matcher for the merge, canvas work to draw and
route it, and a re-proof of the round-trip invariant over the enlarged image.
That is a wave, not a clause.

### A correction to §4, found by writing the table

§4 said `flow.for` raised as `UnsupportedStmt("while")`. It did not; it raised as
`UnsupportedStmt("assign")`. A `for` expansion satisfies the *while* matcher from
its third statement, so `try_raise_while` caught the tail and then choked on the
index increment inside the body. The overlap is now pinned
(`loopshape::a_for_expansion_looks_like_a_while_from_its_third_statement`), and
it is why every consumer tries `match_for` first.

### And a live bug the table found

`raise` **hung** on an action bound to a local. `raise_chain`'s arm for that
shape `continue`d without advancing the statement index, so `let e =
engine.spawn("x")` raised the same statement for ever, adding a graph node per
pass until the process ran out of memory. It was reachable from the editor:
`graph_open_actor` raises every handler of a `.inf_act`, and `engine.spawn` is
the one action in the kit with a consumed data output — the exact shape "spawn
something and keep the entity" produces. Nothing had ever asked `raise` about it,
because every round-trip fixture in the tree starts from a *graph* and none of
them wires a spawn's `entity` onward. Fixed in SCRIPT1a, with the regression
built from a graph so the round-trip invariant applies to it.

**And the audit fixed how that regression fails.** Reverting the `i += 1`
announced itself as a **7.5 GB allocation failure after more than a minute** —
a detection, but the worst kind: an OOM-killed CI job with no line number, a
frozen editor, and on this machine a paging-file incident the repo has a law
about. `raise_chain` now asserts its own advance (one comparison per statement)
and returns `RaiseError::NoProgress(i)` instead, so the same mutation fails the
same arm **immediately**, named, in 0.00 s. That is P21's "a refusal is a value"
applied to a walk rather than to a gameplay node, and it also covers a matcher
that ever returned a zero `consumed`, which nothing does today and nothing
checked.

## A.6 The round trip, stated exactly

Three laws, all in `crates/inf-script/tests/roundtrip.rs`:

1. **`parse(emit(f)) == f`, exactly, for every `f` the parser produces** — ids,
   binding kinds, type annotations and mutability all survive. That is every
   `.infini` file anyone writes. For the committed corpus something stronger
   holds: `emit(parse(src)) == src`, so opening a script and saving it produces
   no diff — **of a file with no comments in it** (A.1: the IR has no trivia, so
   a comment, a blank line and the author's own indentation go on the first
   save; the text is a fixed point from the second).
2. **`emit(parse(emit(f))) == emit(f)`, exactly, for every `f`** — including IR a
   graph produced, whose anonymous locals the lowerer numbered in its own walk.
   The emitter renumbers into the parser's allocation order, so the text is a
   fixed point from the first save.
3. **`parse(emit(f))` runs identically to `f`** — the same host calls in the same
   order with the same arguments, and the same wire values. This is the honest
   replacement for the equality law 1 cannot make about graph-lowered IR, and it
   asserts the program rather than the report.

Two rules the round trip rests on, each with its own arm:

* **A negated literal is folded into a negative one, in both directions.** The
  parser turns `-(1)` into `Lit::Int(-1)` and the emitter prints `Unary(Neg,
  Lit)` as the negative literal. The first draft kept them apart, so that a graph
  holding `Unary(Neg, Lit)` (a `math.neg` node wired to a `lit.float`) could
  print and come back. A.8's bridge then found what that cost:
  `inf_transpile::emit` **refuses** `Unary(Neg, Lit)`, because the lifter folds
  `-lit` on the way back and the shape cannot round-trip through Rust. **A
  language able to write it is a language able to write programs the cook
  refuses**, and that is not a trade this arc makes.
* **A member variable shadowed by a local prints explicitly.** `vars::get("x")`
  is `x` — until a `local x` is live there, at which point it prints
  `var.get("x")`, because a bare `x` would mean the local.

The corpus is checked for coverage before it is trusted: every `Stmt` variant,
every `Expr` variant, every `BinOp`, both `UnOp`s, all four `Lit` kinds and all
three `Binding` kinds, or the meta-arm fails. A round-trip test over the empty
program proves nothing.

**What the emitter refuses** is six IR states, four of which no producer in the
tree makes: a non-finite float literal (`Lit::Float` is documented finite), a
binder whose name is not an identifier, two live binders of one name, and a
handler whose parameters are not its event's signature. The SCRIPT1a audit added
the other two, and they are *resource* and *grammar* bounds rather than shapes:

* `EmitError::TooDeep` — IR nesting deeper than `MAX_EMIT_NESTING` (twice the
  parser's budget, so that anything the parser accepts the emitter can write).
  A graph can chain operator nodes without limit and lowering makes the chain a
  left-deep `Expr`; printing one was a stack overflow in whatever process opened
  the Blueprint as text.
* `EmitError::UnspellableStatement` — a `Stmt::ExprStmt` that is not a call the
  grammar accepts in statement position (a literal, a local, a binary
  expression, a `vars::get`, which prints as a bare name, or a
  `nodestate::get_or`, which has a value spelling and no statement one).
  InfiniScript has no evaluate-and-discard statement. `raise` refuses the same
  shape (`UnsupportedStmt("non-call expr stmt")`), so this is a state both faces
  agree they cannot hold rather than a new limit — and before the audit all five
  printed happily and re-parsed not at all.

And **anything the parser accepts, the emitter can write** — the two reserved
names (`var`, `nodestate`) are refused at every declaration site for exactly that
reason, because a program that parses and then cannot be printed is a worse
failure than a refusal with a line number. The audit made that sentence a
*measurement* rather than a policy: `hostile.rs` mutates a real script one
character at a time (~3 500 inputs) and requires every mutation that parses to
emit and re-parse to identical IR.

**And the converse, which the wave had not stated and which was false**: anything
the emitter writes, the parser must read back. Two shapes broke it and one of
them is two nodes on a canvas —

* **a comparison in a comparison's left operand.** `cmp.lt` wired into
  `cmp.eq`'s `a` input lowers to `Binary(Eq, Binary(Lt, …), …)` and printed
  `1.0 < 2.0 == true`, which the grammar refuses as a chained comparison
  (A.2: `cmp` does not chain). The emitter parenthesised the *right* operand of
  a non-associating operator and not the left; 36 of the 338 operator pairings
  were this. Fixed, and pinned by the full 338-pairing sweep plus the two-node
  graph itself;
* the `ExprStmt` family above.

An emitter that produces a file its own parser rejects is worse than one that
refuses, because the refusal is silent until somebody opens the result — and
"graphs and text are two views of one program" is the claim it falsifies.

## A.7 Containment, at its true granularity

A failing script takes **its handler, on its actor, for that tick**. `run_event`
propagates a `RunError` with `?`, so the failure aborts that handler *at the
failing statement*; `run_on_guid` one level up logs it and the per-actor loop
moves on. Both hosts do this identically.

The bound a designer must know: **a `Tick` that fails after one of three state
writes leaves the actor half-updated.** A cone model would not predict that.
Narrowing it means running scripts through `inf_graph::exec` instead of the IR
interpreter, which is a different design, and this arc does not propose it.

A `while true do … end` cannot hang the editor: `while` and `for` lower to the
counter-guarded expansion, so the bound lives *in the IR* and the interpreter and
the transpiled Rust share it.

**And the other end of the same granularity, added by SCRIPT3 and written down
here by its audit** (the wave cited this section for it and this section did not
say it). `engine.destroy` removes an entity **and everything parented under
it**, and every *actor* among them stops — at the **end** of the handler that
did it, never at the call. So `engine.destroy(var.get("entity"))` followed by a
`debug.print` prints: the statement after a destroy runs, exactly as the
statement after any other call does. The host records the guids and the session
drains them once the handler returns, which is this rule rather than an exception
to it — the unit of both halves is *a handler, on its actor, for that tick*.
Armed by `an_actor_that_destroys_itself_finishes_the_handler_and_stops` and, for
the subtree, `a_destroy_stops_the_handlers_of_everything_under_it`.

## A.8 The third leg: text → IR → Rust → IR

`crates/inf-script/tests/transpile_bridge.rs` closes the triangle. Nothing else
did: `inf-transpile`'s 38 proptests and its hand-edit corpus generate their IR
from graphs and from Rust, and **neither producer makes a `Binding::Raw` binder,
a `Stmt::Assign`, a non-terminal `if`, or a `for` whose index carries a name**.
A text face produces all four the first day somebody writes a script, and the
bridge's own meta-arm fails if the corpus stops carrying them.

Three claims: the transpiler renders every script; `lift` recovers it
**structurally** (no verbatim fallback, no warning), so a designer's script is
editable Rust rather than an opaque blob; and regeneration is byte-idempotent
with the program unchanged.

**It does not compile the generated Rust and it does not run it.** That was
SCRIPT1b's crown gate to build, and it is built: `crown_parity.rs` compiles the
transpiler's output with `rustc`, runs it, and diffs the trace against the
interpreter byte for byte. This file still narrows what that gate has left to
prove — it is the *structural* leg, over IR shapes no graph and no Rust source
produces — and it does not stand in for it. What the crown gate then found about
the generated Rust (an attribute with no macro behind it, and a monomorphic
`vars::get`) is in §8.

**The one literal a script can write and the cook cannot render**: `i64::MIN`.
`-9223372036854775808` is `-(9223372036854775808)` in Rust source and the
magnitude overflows, so `inf_transpile::emit` refuses it by name
(`EmitError::IntMin`). It is not a hole the language opens — a `lit.int` node
holds it just as happily — and the interpreter computes with it perfectly, so a
script using it **previews and does not cook**. Pinned by
`the_one_literal_a_script_can_write_and_the_cook_cannot_render` so SCRIPT1b's
cook reports it as an advisory rather than a stack trace.

## A.9 What v1 deliberately does not have

Named by scope rather than discovered: no tables, arrays or structs (the IR's
value set is `Float`/`Int`/`Bool`/`Str`), no user-defined types, no closures, no
`repeat`/`break`/`continue`, no string concatenation operator (`debug.print`
takes one string), no `for … in` over a collection, no modules or `require`, no
multiple return values, and no varargs.

**RETIRED by wave SCRIPT2a: "no calls from a handler to its unit's own
`function` declarations".** This entry read *"the IR has no user-function call
form, and adding one is an IR change with the P6 vars-via-Host bar to clear"* —
and the bar was cleared by finding that **no IR change was needed**. A one-segment
`Expr::Call` is the whole feature (§8, A.2, A.4). It is left standing here as the
list's own worked example, because the prediction it makes is the one this list
is *for*: every remaining line is a pricing question about the IR, and the first
question to ask of one is not "what does the IR need" but **"what can the shapes
it already has not say"**. Two of the entries above have been answered that way
now — member variables at P6, the call form here.

The call form arrived with two bounds of its own, both refusals rather than
surprises: **no recursion**, and **no call chain longer than
`MAX_CALL_DEPTH`** (64), because the interpreter bounds a chain and the
transpiled Rust does not, so a program past either would run in one face and be
refused by the other.

Every remaining one is a *pricing* question about the IR, not about the parser.
That is the design working: the language cannot outgrow the execution model by
accident.
