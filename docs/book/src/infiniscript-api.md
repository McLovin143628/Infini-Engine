<!-- GENERATED from the InfiniScript verb registry by `cargo test -p inf-script --test api_manual`. Do not hand-edit; re-bless with INF_BLESS_API_MANUAL=1. -->

# The InfiniScript API

Every verb an InfiniScript can call, generated from the engine's own verb registry — the same table the Blueprint palette is built from and the same one the parser resolves a call against. If it is not here, a script cannot say it.

The surface is **132 registered nodes across 26 namespaces**. Not all of them are *callable*: the arithmetic and comparison operators, the control-flow palette, the literals, the events and the two member-variable nodes are all written as syntax instead, and the last section lists each with the spelling that replaces it.

**How to read a row.** `door.use(x: float, y: float, z: float)` takes three arguments in that order. A name in `[square brackets]` is optional and defaults to zero, false or the empty string. A verb marked *statement* runs for its effect and is written on a line of its own; a verb marked *value* answers something and can go inside an expression. A verb that is both can be used either way — write it as a statement to ignore what it answers.

## `math.*`

| call | kind | answers | what it does |
|---|---|---|---|
| `math.abs([a: float])` | value | `out`: float | The magnitude of a number, dropping its sign. Stays an integer when its argument is one. |
| `math.floor([a: float])` | value | `out`: float | The largest whole number no greater than x — rounds toward negative infinity, so -1.5 floors to -2. |
| `math.ceil([a: float])` | value | `out`: float | The smallest whole number no less than x — rounds toward positive infinity. |
| `math.round([a: float])` | value | `out`: float | The nearest whole number, with halves rounded away from zero. |
| `math.sqrt([a: float])` | value | `out`: float | The square root of x. Not a number for a negative x, which the rest of the program carries as a value rather than a failure. |
| `math.sin([a: float])` | value | `out`: float | The sine of an angle in RADIANS, through the engine's own portable polynomial — bit-identical on every operating system, which the platform library is not. |
| `math.cos([a: float])` | value | `out`: float | The cosine of an angle in RADIANS, through the engine's own portable polynomial. |
| `math.min([a: float], [b: float])` | value | `out`: float | The smaller of two numbers. Absorbs a not-a-number argument and answers the other one. |
| `math.max([a: float], [b: float])` | value | `out`: float | The larger of two numbers. Absorbs a not-a-number argument and answers the other one. |
| `math.pow([a: float], [b: float])` | value | `out`: float | a raised to the power b. NOT bit-portable across operating systems — the one hole in the math palette, and the reason a committed value should not depend on it. |
| `math.clamp([x: float], [min: float], [max: float])` | value | `out`: float | Constrain x to [min, max] (non-panicking; inverted range yields max). |
| `math.lerp([a: float], [b: float], [t: float])` | value | `out`: float | Linear interpolation a + (b − a) · t (unclamped t). |
| `math.to_int([a: float])` | value | `out`: int | Truncate a Float toward zero to an Integer (saturating). |
| `math.to_float([a: int])` | value | `out`: float | Widen an Integer to a Float. |

## `engine.*`

| call | kind | answers | what it does |
|---|---|---|---|
| `engine.set_rotation(angle: float)` | statement | — | Turn the acting entity to an absolute yaw, in DEGREES. NOT IMPLEMENTED BY EITHER HOST today: the call is logged and does nothing, which is what every unrecognised call does. |
| `engine.spawn(prefab: string)` | statement, value | `entity`: int | Place a copy of a named prefab in the world and report its entity id. The name is the one string in this whole kit the cook resolves as an ASSET, so a name matching nothing is a blocking advisory. NOT IMPLEMENTED BY EITHER HOST today: the call is logged, nothing is spawned, and the reported id is unusable. |
| `engine.destroy(entity: int)` | statement | — | Remove an entity from the world. NOT IMPLEMENTED BY EITHER HOST today: the call is logged and nothing is removed. |

## `debug.*`

| call | kind | answers | what it does |
|---|---|---|---|
| `debug.print(message: string)` | statement | — | Write a line to the Output Log, tagged with the actor that wrote it. The one verb that is always safe to add: it changes nothing a replay can see. |

## `dispatch.*`

| call | kind | answers | what it does |
|---|---|---|---|
| `dispatch.call(target: int, name: string)` | statement | — | Fire the custom event `name` on `target` (and its bound listeners). |
| `dispatch.bind(source: int, name: string, handler: string)` | statement | — | Subscribe this actor's `handler` custom event to `source`'s `name` event. |
| `dispatch.unbind(source: int, name: string, handler: string)` | statement | — | Remove this actor's `handler` subscription to `source`'s `name` event. |

## `physics2d.*`

| call | kind | answers | what it does |
|---|---|---|---|
| `physics2d.move_and_slide(entity: int, [motion_x: float], [motion_y: float])` | statement, value | `grounded`: bool | Slide an entity by a motion vector, resolving collisions. |
| `physics2d.is_grounded(entity: int)` | value | `grounded`: bool | Whether the entity is currently touching the ground. |
| `physics2d.raycast.<hit>([origin_x: float], [origin_y: float], [dir_x: float], [dir_y: float], [max: float])` | value | `hit`: bool, `point_x`: float, `point_y`: float, `normal_x`: float, `normal_y`: float | Cast a ray; reports hit + world point + surface normal. |
| `physics2d.set_velocity(entity: int, [vx: float], [vy: float])` | statement | — | Set a body's linear velocity outright, in metres per second. It overwrites whatever the solver had; use Apply Impulse when the motion should add to what is already there. |
| `physics2d.get_velocity.<x>(entity: int)` | value | `x`: float, `y`: float | A body's linear velocity in metres per second. Two results, so a call names the one it wants: `physics2d.get_velocity.x(e)`. |
| `physics2d.apply_impulse(entity: int, [vx: float], [vy: float])` | statement | — | Add momentum to a body, in newton-seconds — a kick rather than a speed. Mass matters: the same impulse moves a light body further. |

## `physics3d.*`

| call | kind | answers | what it does |
|---|---|---|---|
| `physics3d.move_and_slide(entity: int, [motion_x: float], [motion_y: float], [motion_z: float])` | statement, value | `grounded`: bool | Slide an entity by a 3D motion vector, resolving collisions. |
| `physics3d.is_grounded(entity: int)` | value | `grounded`: bool | Whether the entity is currently touching the ground. |
| `physics3d.raycast.<hit>([origin_x: float], [origin_y: float], [origin_z: float], [dir_x: float], [dir_y: float], [dir_z: float], [max: float])` | value | `hit`: bool, `point_x`: float, `point_y`: float, `point_z`: float, `normal_x`: float, `normal_y`: float, `normal_z`: float | Cast a 3D ray; reports hit + world point + surface normal. |
| `physics3d.set_velocity(entity: int, [vx: float], [vy: float], [vz: float])` | statement | — | Set a body's linear velocity outright, in metres per second, on all three axes. |
| `physics3d.get_velocity.<x>(entity: int)` | value | `x`: float, `y`: float, `z`: float | A body's linear velocity in metres per second. Three results, so a call names the one it wants: `physics3d.get_velocity.y(e)`. |
| `physics3d.apply_impulse(entity: int, [vx: float], [vy: float], [vz: float])` | statement | — | Add momentum to a body, in newton-seconds — a kick rather than a speed. Mass matters: the same impulse moves a light body further. |

## `input.*`

| call | kind | answers | what it does |
|---|---|---|---|
| `input.is_down(key: string)` | value | `down`: bool | True while the named action/key is held. |
| `input.just_pressed(key: string)` | value | `pressed`: bool | True only on the tick the named action/key went down (rising edge). |

## `audio.*`

| call | kind | answers | what it does |
|---|---|---|---|
| `audio.play(entity: int)` | statement | — | Start (or restart) the entity's AudioSource clip. |
| `audio.stop(entity: int)` | statement | — | Stop the entity's currently-playing AudioSource voice. |
| `audio.set_volume(entity: int, [volume: float])` | statement | — | Set the entity's AudioSource base volume (linear). |
| `audio.set_pitch(entity: int, [pitch: float])` | statement | — | Set the entity's AudioSource pitch (playback-rate factor). |

## `sky.*`

| call | kind | answers | what it does |
|---|---|---|---|
| `sky.get_time_of_day()` | value | `seconds`: float | The level clock, in UTC seconds since midnight (0..86400). |
| `sky.set_time_of_day([seconds: float])` | statement | — | Set the level clock, in seconds since midnight (wraps at 86400). |
| `sky.get_rate()` | value | `rate`: float | How fast the clock runs: simulated seconds per second (0 = frozen). |
| `sky.set_rate([rate: float])` | statement | — | Set how fast the clock runs (0 freezes it; negative runs it backwards). |
| `sky.set_weather(preset: string, [blend_seconds: float])` | statement | — | Blend the weather to a preset (clear/overcast/storm/fog/snow) over `blend_seconds` (0 = instantly; negative = the level's authored blend time). |
| `sky.get_weather()` | value | `preset`: string | The weather preset the level is in (or blending toward). |
| `sky.get_precipitation()` | value | `intensity`: float | How hard it is raining or snowing right now, 0..1 (0 = dry). |
| `sky.get_wind_speed()` | value | `speed`: float | Wind speed in metres per second — what drifts the clouds and slants the rain. |
| `sky.is_day()` | value | `day`: bool | True while the sun is above the horizon. The same test the atmosphere uses to choose a sun or a moon, so it flips exactly when the sky does. |
| `sky.get_hour()` | value | `hour`: float | The level clock as an hour in 0..24, local to the level's longitude and time zone — the same number the crowd's daily schedules run on. Use this rather than dividing Get Time of Day, which is UTC. |
| `sky.get_cloud_coverage()` | value | `coverage`: float | How much of the sky the clouds cover, 0..1, including a blend in progress. 0 with weather off. |
| `sky.get_fog_density()` | value | `density`: float | Height-fog extinction in inverse metres — visibility is roughly 3 / density. 0 with weather off, and a fog preset is about 0.02 (150 m). |

## `water.*`

| call | kind | answers | what it does |
|---|---|---|---|
| `water.is_in_water(entity: int)` | value | `in_water`: bool | True while any part of the entity is under a water surface. Instantaneous: use the On Enter/Exit Water events for debounced edges. |
| `water.surface_height([x: float], [z: float])` | value | `height`: float | World Y of the highest water surface over (x, z) right now — 0 where there is no water. |
| `water.submerged_fraction(entity: int)` | value | `fraction`: float | How much of the entity is under water, 0..1 (0 = dry, 1 = fully under). |

## `voxel.*`

| call | kind | answers | what it does |
|---|---|---|---|
| `voxel.carve_sphere(entity: int, [x: float], [y: float], [z: float], [radius: float])` | statement, value | `removed_m3`: float | Dig a ball out of the entity's voxel volume; reports the cubic metres removed (0 if the volume's Runtime Carve is off). |
| `voxel.carve_box(entity: int, [x: float], [y: float], [z: float], [half_x: float], [half_y: float], [half_z: float])` | statement, value | `removed_m3`: float | Dig an axis-aligned box out of the entity's voxel volume; reports the cubic metres removed. Extents are HALF-extents, in metres. |
| `voxel.fill_sphere(entity: int, [x: float], [y: float], [z: float], [radius: float], [material: int])` | statement, value | `added_m3`: float | Add a ball of solid material to the entity's voxel volume; reports the cubic metres added. Material is a splat-layer index (0..3). |
| `voxel.is_solid([x: float], [y: float], [z: float])` | value | `solid`: bool | True when any of the level's voxel volumes has rock at that world point. Says nothing about the heightfield — solid ground reads false. |
| `voxel.ground_height([x: float], [z: float])` | value | `height`: float | World Y of the topmost VOXEL surface over (x, z) — 0 where no volume answers. For the ground a character stands on, use Terrain Height At, which combines the heightfield with this. |

## `destruct.*`

| call | kind | answers | what it does |
|---|---|---|---|
| `destruct.apply_damage(entity: int, [energy_j: float])` | statement, value | `absorbed_j`: float | Spend energy breaking an actor's fracture bonds, then let whatever that leaves unsupported collapse. Energy is in JOULES; reports the joules actually absorbed (0 if the actor's Runtime Destruct is off, it has no fracture data, or the blow was too weak). |
| `destruct.radial_impulse([x: float], [y: float], [z: float], [impulse_ns: float], [radius_m: float])` | statement, value | `bodies_hit`: float | Push every dynamic body within a radius away from a world point — an explosion. Impulse is NEWTON-SECONDS delivered at 1 m, falling off as the inverse square beyond that; radius is in metres. Reports how many bodies were pushed. Breaks nothing on its own: pair it with Apply Damage. |
| `destruct.is_intact(entity: int)` | value | `intact`: bool | True while none of the actor's fracture chunks has come off. False for an actor with no fracture data at all — nothing broken, but nothing that CAN break either; ask Chunk Count to tell those apart. |
| `destruct.chunk_count(entity: int)` | value | `chunks`: int | How many chunks the actor's fracture has — 0 when it has none, which is how you tell 'indestructible' from 'already broken'. |

## `item.*`

| call | kind | answers | what it does |
|---|---|---|---|
| `item.define(toml: string)` | statement, value | `count`: int | Add item definitions to the session's catalogue, as name-keyed TOML. Each table is one item id; `label`, `stack_max` and `mass_kg` are optional, and a `[<id>.weapon]` sub-table makes it a weapon (damage_j, rounds_per_minute, magazine, reload_s, spread_deg, range_m, automatic, kind). Reports how many definitions were taken; a malformed document takes NONE and logs why. |
| `item.spawn_pickup(id: string, [x: float], [y: float], [z: float], [count: int])` | statement, value | `ok`: bool | Put an item on the ground as an entity the interact key can pick up. The id must already be in the catalogue (Define Items); an unknown id spawns nothing and says so. |
| `item.give(entity: int, id: string, [count: int])` | statement, value | `left`: int | Put items straight into an actor's inventory, creating one if it has none. Reports how many did NOT fit — 0 means all of them did. |
| `item.equip(entity: int, id: string)` | statement, value | `ok`: bool | Equip an item the actor is already carrying, and give it a full magazine if it is a weapon. False when the actor does not have one. |
| `item.count(entity: int, id: string)` | value | `count`: int | How many of an item the actor is carrying, across every slot. |

## `door.*`

| call | kind | answers | what it does |
|---|---|---|---|
| `door.spawn(toml: string)` | statement, value | `count`: int | Hang doors in the world, as name-keyed TOML. Each table is one door: `hinge = [x, y, z]` at the leaf's mid-height, `closed_yaw_deg` toward the free edge, `inside_yaw_deg` for the face the lock is on, plus the optional `width_m`, `height_m`, `thickness_m`, `open_limit_deg`, `locked` and `label`. Reports how many were hung. The building grammar hangs its own without this. |
| `door.is_open([x: float], [y: float], [z: float])` | value | `open`: bool | True when the door nearest a world point is open far enough to walk through. False when there is no door within a few metres of it. |
| `door.is_locked([x: float], [y: float], [z: float])` | value | `locked`: bool | True when the door nearest a world point is bolted. False when there is no door within a few metres, and false for a lock that has been broken open — a bolt that holds nothing is not a locked door. |
| `door.use([x: float], [y: float], [z: float])` | statement, value | `moved`: bool | Open or close the door nearest a world point — the E key, as a verb. Reports whether the leaf actually started to move: a locked door, a door with unusable numbers and no door at all all report false, which is the same thing the prompt tells a player. |
| `door.lock([x: float], [y: float], [z: float])` | statement, value | `changed`: bool | Throw or release the bolt on the door nearest a world point, as if a character standing there turned a key. Refused on an OPEN leaf (a door standing open with its bolt thrown is a lock nobody can see) and from the wrong face. Reports whether the bolt moved. |

## `health.*`

| call | kind | answers | what it does |
|---|---|---|---|
| `health.set(entity: int, [joules: float])` | statement, value | `ok`: bool | Give an actor a body worth this many JOULES — what it can absorb before it stops working and goes limp. This engine has no hit points: a bullet, a kick and a collapsing wall are all energy, which is why there is no conversion to tune. A rifle round is about 1 700 J. |
| `health.get(entity: int)` | value | `joules`: float | How many joules the actor can still absorb. 0 for an actor with no health at all, which is every actor nothing has given one to. |
| `health.damage(entity: int, [joules: float])` | statement, value | `absorbed`: float | Take energy out of an actor's body, in JOULES — a bullet, a kick, a falling wall. Reports how much was actually absorbed, which is less than asked for when the blow finished the actor off, and 0 for an actor with no health at all. Downing is what happens at zero; there are no hit points to convert. |
| `health.fraction(entity: int)` | value | `fraction`: float | How much of its body an actor has left, 0..1, against the amount it started with. 0 for an actor with no health at all — pair it with Get Health when the difference matters. |
| `health.is_downed(entity: int)` | value | `downed`: bool | True once an actor has absorbed everything it can and gone limp. False for an actor with no health at all, which is not the same as unhurt and is the honest answer to `is it down`. |

## `ik.*`

| call | kind | answers | what it does |
|---|---|---|---|
| `ik.set_goal(entity: int, [goal: int], [x: float], [y: float], [z: float])` | statement, value | `ok`: bool | Move one of this entity's authored IK goals to a WORLD point. `goal` is its index in the IK Target component. Reports false when there is no such goal. |
| `ik.set_goal_weight(entity: int, [goal: int], [weight: float])` | statement, value | `ok`: bool | How much of one authored goal's solve to apply, 0..1 — the door for fading a foot plant in and out instead of snapping. Reports false when there is no such goal. |
| `ik.enable_goal(entity: int, [goal: int], [enabled: bool])` | statement, value | `ok`: bool | Turn one authored IK goal on or off. A disabled goal is not solved and costs nothing in the trace. Reports false when there is no such goal. |
| `ik.reached(entity: int)` | value | `reached`: bool | True when EVERY one of this entity's IK goals landed on its target last step. False when it has none, or when any chain refused. |
| `ik.reach_error(entity: int)` | value | `error_m`: float | How far the worst of this entity's IK tips missed by last step, METRES. 0 when nothing was solved — from gameplay's side that is the same fact as a perfect solve. |

## `anim.*`

| call | kind | answers | what it does |
|---|---|---|---|
| `anim.set_param(entity: int, name: string, [value: float])` | statement, value | `ok`: bool | Set a named parameter on this entity's animation state machine. It SHADOWS an actor variable of the same name and persists until set again. Reports false for an entity with no state machine, or for a value that is not a number. |
| `anim.set_trigger(entity: int, name: string)` | statement, value | `ok`: bool | Arm a declared trigger parameter once. Unlike Set Anim Parameter this is an EVENT: arming twice on consecutive steps fires twice, and an armed trigger no transition consumes stays armed. Reports false for an entity with no state machine. |
| `anim.query_state(entity: int, name: string)` | value | `active`: bool | True while this entity's state machine is in the named state. False for an entity that has never stepped one — a state nothing has entered is not a state anything is in. |
| `anim.consume_notify(entity: int, name: string)` | statement, value | `fired`: bool | TAKE one of this step's animation notifies by name — a state's enter/exit event or a clip's event marker (a footstep). True exactly once per fired name per step: two handlers racing for one footstep get one. |

## `terrain.*`

| call | kind | answers | what it does |
|---|---|---|---|
| `terrain.height_at([x: float], [z: float])` | value | `height`: float | The ground height in metres at a world XZ — the terrain heightfield with any voxel surface above or below it folded in, which is the same number the character controller stands on. Answers 0 where the level has no ground, the way `water.surface_height` does, because the IR has no optional Float. |

## `crowd.*`

| call | kind | answers | what it does |
|---|---|---|---|
| `crowd.population()` | value | `count`: int | How many crowd agents the level is simulating right now, across every tier — the dormant ones included, because a schedule keeps running for an agent nobody can see. |
| `crowd.blocked()` | value | `count`: int | How many agents are stuck against something they cannot walk through. A number to watch rather than to act on: a large one means the level's navigation is not joined up. |
| `crowd.homes()` | value | `count`: int | How many homes the level's buildings have offered the society — the people it has, plus the ones still waiting for a day, plus the ones the population ceiling declined. |
| `crowd.workplaces()` | value | `count`: int | How many workplaces the level's buildings have offered the society. |

## `zone.*`

| call | kind | answers | what it does |
|---|---|---|---|
| `zone.contains(entity: int, [x: float], [y: float], [z: float], [half_x: float], [half_y: float], [half_z: float])` | value | `inside`: bool | True when the actor's collider overlaps an axis-aligned box, given as a centre and half-extents in metres. False for an entity with no collider — this asks the physics world, so an actor the physics world has never heard of is not anywhere. |
| `zone.count([x: float], [y: float], [z: float], [half_x: float], [half_y: float], [half_z: float])` | value | `count`: int | How many distinct entities have a collider overlapping the box. Counts ENTITIES, not colliders: an actor with three colliders in the box is one, and a collider the physics world cannot name is none. It counts EVERYTHING with a collider — the ground's heightfield and a door's leaf included — so a box on the ground reads higher than the number of actors in it. Use Is In Zone when the question is about a particular actor. |

## Written as syntax instead

These are registered nodes a Blueprint graph draws and InfiniScript spells another way. Calling one is a refusal that names the replacement, so the compiler tells you this table rather than making you find it.

| node | write instead |
|---|---|
| `lit.float` | `lit.float` is a literal — write the value itself, like `1.5` |
| `lit.int` | `lit.int` is a literal — write the value itself, like `1.5` |
| `lit.bool` | `lit.bool` is a literal — write the value itself, like `1.5` |
| `lit.str` | `lit.str` is a literal — write the value itself, like `1.5` |
| `math.add` | `math.add` is the `+` operator in InfiniScript — write `a + b` |
| `math.sub` | `math.sub` is the `-` operator in InfiniScript — write `a - b` |
| `math.mul` | `math.mul` is the `*` operator in InfiniScript — write `a * b` |
| `math.div` | `math.div` is the `/` operator in InfiniScript — write `a / b` |
| `math.rem` | `math.rem` is the `%` operator in InfiniScript — write `a % b` |
| `math.neg` | `math.neg` is unary minus — write `-(a)` |
| `cmp.eq` | `cmp.eq` is the `==` operator in InfiniScript — write `a == b` |
| `cmp.ne` | `cmp.ne` is the `~=` operator in InfiniScript — write `a ~= b` |
| `cmp.lt` | `cmp.lt` is the `<` operator in InfiniScript — write `a < b` |
| `cmp.le` | `cmp.le` is the `<=` operator in InfiniScript — write `a <= b` |
| `cmp.gt` | `cmp.gt` is the `>` operator in InfiniScript — write `a > b` |
| `cmp.ge` | `cmp.ge` is the `>=` operator in InfiniScript — write `a >= b` |
| `logic.and` | `logic.and` is the `and` operator in InfiniScript — write `a and b` |
| `logic.or` | `logic.or` is the `or` operator in InfiniScript — write `a or b` |
| `logic.not` | `logic.not` is the `not` operator — write `not a` |
| `var.get` | `var.get` is a member variable — write its name, or `var.get("…")` when the name is not an identifier |
| `var.set` | `var.set` is a member variable — write its name, or `var.get("…")` when the name is not an identifier |
| `flow.branch` | `flow.branch` is control flow, which InfiniScript writes as syntax — `if … then … end` |
| `flow.sequence` | `flow.sequence` is control flow, which InfiniScript writes as syntax — statements simply follow one another |
| `flow.return` | `flow.return` is control flow, which InfiniScript writes as syntax — `return` |
| `flow.while` | `flow.while` is control flow, which InfiniScript writes as syntax — `while … do … end` |
| `flow.for` | `flow.for` is control flow, which InfiniScript writes as syntax — `for i = first, last do … end` |
| `flow.do_once` | `flow.do_once` is control flow, which InfiniScript writes as syntax — guard it with a member variable (`if not fired then … end`) |
| `flow.flip_flop` | `flow.flip_flop` is control flow, which InfiniScript writes as syntax — guard it with a member variable |
| `flow.gate` | `flow.gate` is control flow, which InfiniScript writes as syntax — guard it with a member variable |

Events are the other family that is not a call: an event is a handler's header (`on tick(dt) … end`), not something a script invokes.
