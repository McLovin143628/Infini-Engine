# Blueprints 101

**Infini Blueprints** are the visual way to write gameplay. A Blueprint is a node graph that the
engine transpiles into real Rust source — and lifts back from Rust when you hand-edit it — so a
graph and its generated code stay bidirectionally in sync. This is the engine's signature feature:
*two ways to code, one truth.*

## The graph is the source of truth

A Blueprint class is stored as an `.inf_act` (Actor Assembly) asset: it holds the component layout,
default values, and the graph document, plus links to the generated Rust. The doctrine is that the
**graph is authoritative** and Rust is a projection of it. When you hand-edit the generated Rust,
the file watcher parses the changed functions and lifts them back into graph nodes by matching
their canonical shape; anything outside the liftable subset becomes an opaque **snippet node** that
preserves your source verbatim (comments and all) — so a sync never fails and never loses data.

## Author, run, iterate

Open a Blueprint from a `.inf_act` asset to get the **Blueprint** panel — a node canvas built on
@xyflow with exec pins (the white execution wires) and typed data pins (colored per type). Right-
click the canvas for a searchable, sectioned node menu. Blueprints are event-driven: **BeginPlay**
runs once when the actor spawns, **Tick** runs every frame, and there are input, collision, and
custom events. Variables and functions live in side panels; a function library can be shared as an
`.inf_fn` asset.

The classic first Blueprint is "rotate on tick": from the **Tick** event, read the actor's
Transform, add to its yaw, and write it back. Because the editor evaluates your graph with a
tree-walking **interpreter** over the very same IR the transpiler emits, you can watch it run live
in **Simulate** mode without compiling anything — and be certain the compiled, shipped version
behaves identically, because interpreter-vs-compiled parity is a CI gate.

## Read and edit the generated Rust

Any node can jump you to its generated Rust ("Open generated Rust"), which lives in your project's
own cargo workspace so rust-analyzer treats it as first-class — completions, diagnostics, and
go-to-definition all work in the embedded **Code Editor** panel. Edit within the liftable subset
and the graph updates on save; edit outside it and that region becomes a snippet node. Round-trip
correctness (graph → Rust → graph) is enforced by permanent proptests in CI.

## The same graph, as text

A Blueprint is also an **[InfiniScript](./infiniscript.md)** file. `.infini` text and a node
graph are not two programs that agree — they are one program with two editors pointed at it,
because the graph lowers to the same gameplay IR the text parses into. Write whichever suits the
job; a designer who prefers typing never has to draw a wire, and a designer who prefers wires
never has to type.

The text face is the *complete* one: a few shapes a text author writes naturally — an `if` that
is not the last thing in its block, an assignment to a local, a call to one of the file's own
functions — have no node to become, so such a handler opens as text and not as a graph. The
[InfiniScript page](./infiniscript.md) says which.

## When to reach for Rust instead

Blueprints, InfiniScript and Rust are interchangeable, so use whichever fits: prototype and wire
up gameplay events visually or in text, then drop into hand-written Rust for hot loops, complex
algorithms, or heavy data work. There is deliberately **no foreign scripting language** (no
Lua/Rhai) — InfiniScript is a text face on the engine's own IR rather than a second execution
model, so it inherits the interpreter's no-recompile iteration *and* the parity gate that pins it
to the compiled Rust. Safe end-user extensibility is served by WASM mods (see
[Modding](./modding.md)).
