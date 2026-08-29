# InfiniScript

**InfiniScript is the engine's scripting language.** A `.infini` file is plain
text you can read, diff and merge, and it is *the same program* as a Blueprint
graph — not a translation of one. Open a Blueprint as text, edit it, and it
comes back as a graph. Draw the graph and read it as text. There is one program
with two editors pointed at it.

It exists because a designer should change a line and see it, not wait on a
compiler. Saving a `.infini` file re-compiles it and swaps it into the running
Simulate in well under a second, with no `rustc` anywhere on the path.

```lua
actor "Lamp"

var on: bool = false
var brightness: float = 0.0

on begin_play()
    debug.print("lamp ready")
end

on tick(dt)
    if on then
        brightness = math.min(brightness + dt * 4.0, 1.0)
    else
        brightness = math.max(brightness - dt * 4.0, 0.0)
    end
end

on input "use"(pressed)
    if pressed then
        on = not on
    end
end
```

Every verb you can call is in **[The InfiniScript API](./infiniscript-api.md)**,
which is generated from the engine's own registry — if it is not in that page,
no script can say it.

## Where a script lives

**`Content/Scripts/`.** A script is content: it takes a GUID and a sidecar like
any other asset, and an actor in a level binds to it exactly the way it binds to
a Blueprint. Every new project scaffolds `Content/Scripts/Example.infini`.

That is a *convention*, not a lookup — the asset scanner recurses, so a `.infini`
anywhere under `Content/` is found the day it exists, and you may keep a script
beside the thing it drives without telling the engine anything.

> **One rule for a repository.** A script's identity is a GUID recorded in the
> `.infini.toml` sidecar the editor writes the first time it sees the file. Until
> that sidecar exists — a project freshly created by `inf new` and cooked before
> it was ever opened — the GUID is derived from the file's *bytes*, so a checkout
> that rewrote the line endings would rename the script and every binding to it
> would point at nothing. `inf new` therefore writes a `.gitattributes` carrying
> `*.infini -text`, which tells git to leave the bytes exactly as committed. If
> you add InfiniScript to a project that predates this, add that line yourself.

## The shape of a file

A file describes one actor class:

* an optional `actor "Name"` header, which must be the first line;
* `var` declarations — the actor's member variables, with a type and a default;
* `on <event>(…) … end` handlers;
* `function name(…) [-> type] … end` declarations, which handlers can call.

```lua
actor "Turret"

var range_m: float = 30.0 exposed

on tick(dt)
    local target_y = aim_height(range_m)
    engine.set_rotation(target_y)
end

function aim_height(range: float) -> float
    return math.min(range * 0.5, 12.0)
end
```

`exposed` makes a variable editable per-instance in the Details panel.

### Events

| header | fires | receives |
|---|---|---|
| `on begin_play()` | once, when the actor enters play | — |
| `on tick(dt)` | every fixed step | `dt`: seconds |
| `on input "jump"(pressed)` | a named input action changed | `pressed`: bool |
| `on collision(other)` | a collision began | `other`: entity id |
| `on custom "name"()` | another actor dispatched this event | — |
| `on water_enter(water, speed)` / `water_exit` / `water_splash` | a water surface was crossed | the body, and how fast |
| `on destroyed(chunks)` | every one of this actor's chunks came off | how many |

### Statements

```lua
local x = 1.0                  -- a local, typed by what it is given
x = x + 1.0                    -- assignment
speed = 4.0                    -- a bare name is a MEMBER VARIABLE
if a then … elseif b then … else … end
while cond do … end
for i = 1, 10 do … end
return value                   -- the value must be on the SAME line
debug.print("hello")           -- a call
rust [[ let _ = 1; ]]          -- an escape hatch: opaque Rust, not interpreted
```

Comments run from `--` to end of line. **They do not survive a round trip** —
the text and the graph are two views of one program and the program has nowhere
to keep them, so opening a script through the graph editor and saving it drops
its comments and its blank lines.

### Names

A bare identifier resolves innermost-first: a `local` in scope, then a handler
parameter, then — for anything else — **a member variable**. Nothing has to be
declared before it is used, and a name that matches no `var` is a *warning*
rather than an error, because the runtime already refuses an unknown variable by
name.

Shadowing is refused. Two live bindings of one name would print as one name, and
a script that means something different after a save is worse than a rename.

Use `var.get("has space")` / `var.set("has space", v)` when a variable's name is
not an identifier.

### Calls

A call names a namespace and a verb: `door.use(x, y, z)`. Names resolve at
compile time against the engine's registry, which is what makes the surface
safe — **there is no way to reach the platform's maths or files from a script;
you can only call a verb.** A multi-result query names the result it wants:
`physics2d.raycast.hit(…)`.

A handler may also call the file's own `function` declarations, by bare name, in
any order — a handler written above its functions still finds them.

**There is no recursion.** A function that can reach itself, directly or through
another, is refused at compile time with the route it found. Write the repetition
as a `while` or a `for`, which are bounded in a way both the preview and the
shipped build share.

### Types

`float`, `int`, `bool`, `string`. That is the whole set: there are no tables,
arrays, structs, user types or closures in v1. Each of those is a question about
the engine's gameplay IR rather than about the parser, which is the design
working — the language cannot outgrow the execution model by accident.

## Editing a script

Today a `.infini` file is edited in any text editor, and the engine's own editor
watches the folder: **save the file and the running Simulate picks it up**.
Measured, edit to running is 250–370 ms, almost all of it the watcher's debounce
and the editor's drain interval — the engine's own half (read, compile, swap, one
fixed step) is 0.35–0.45 ms, and no `rustc` runs at all. A broken edit never
becomes a program —
the previous one keeps running and the diagnostics, with their line and column,
appear in the **Output Log**.

Two things a designer should expect from a hot reload:

* **state survives.** Member variables keep their values, and a variable the edit
  *added* is seeded from its default.
* **changing a default does not change a running actor.** A variable that
  already has a value keeps it; the new default reaches the next actor to be
  created.

> A `.infini` language mode inside the engine's own code editor — highlighting,
> and diagnostics in the Problems panel — is the next wave's work. Until then the
> Output Log is where a script's errors appear.

## What happens at cook time

The cook lowers each script to the engine's gameplay IR and packs that, and the
shipped player interprets it — the same road a Blueprint travels, so a script
inherits every guarantee that road already has, including the one this engine
cares most about: **what you previewed is what ships**, compared step by step and
byte for byte.

Scripts can also be *transpiled* to real Rust — that is what the Code tab shows —
and the two are gated equal to each other by compiling the generated program and
diffing its trace against the interpreter's. Two known limits of that path today:
a member variable that is not a float, and a `string` parameter on one of your own
functions, do not transpile. Both interpret perfectly and both ship.

One literal previews and does not transpile: the most negative integer,
`-9223372036854775808`. The cook reports it as an advisory naming the handler.
