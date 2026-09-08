# Animation

Infini Engine drives skeletal characters with clips, blend spaces, and state machines, all wired
to gameplay through Blueprints. The `samples/character-demo` project is the reference: an
idle/run/jump character that walks and jumps across terrain, driven entirely by a Blueprint.

## Skeletons and clips

A character carries a **SkeletalMesh** component that references a skeleton asset (`.inf_skel`) and,
optionally, a skinned mesh. Animation clips are `.inf_anim` assets — the character demo ships three
programmatic ones: an idle bob, a forward-moving run with **root motion**, and a jump arc. Root
motion means the clip drives the actor's world position from the animation itself, so locomotion
stays in sync with the feet.

## State machines

An **AnimStateMachine** component references a state-machine asset (`.inf_sm`) that decides which
clip plays. The demo's `Locomotion.inf_sm` has three states — idle, run, jump — with transitions:
idle → run when speed exceeds a threshold, run → idle when it drops back, and any-state → jump on a
jump trigger with an exit back to locomotion. The state machine reads its parameters from Blueprint
variables (`params_from_vars`), so gameplay code sets `speed` and `jump` and the animation system
reacts.

## Driving it from a Blueprint

The character's `.inf_act` Blueprint ties input to motion. On **Tick** it reads input actions
(left/right, jump), applies gravity, integrates a tracked position, clamps the character to the
terrain height beneath it, and moves the body with a character-controller **move-and-slide** call —
then writes `speed` and `jump` for the state machine to consume. Because the same interpreter that
previews the graph also runs in play-in-editor and (as compiled Rust) in the shipped player, the
animation you tune in the editor is the animation that ships. The runtime gate for this sample
scripts input and asserts the character crosses the terrain, jumps and lands, and transitions
idle → run → jump — with play-in-editor proven byte-identical to shipping.

## Editing in the viewport

Select the character to see its components in Details; the state machine and animation references
are asset-ref pickers, and the Blueprint variables that feed the state machine are editable there
too. Use **Simulate** (the play cluster's dropdown) to tick physics and animation without full game
logic, which is the fastest way to check a transition or a blend without leaving the editor.

## Taking cover

A character can put its back against the thing in front of it, and **the object decides the
stance**: a car's flank or a counter is crouch cover, a wall or a container is stand cover. This is
`MovementMode::Cover`, and it is one input verb plus one probe.

### The verb

| control | action | what it does |
|---|---|---|
| **T** (keyboard) | `cover` | take cover behind what is in front, or leave it |
| **D-pad down** (gamepad) | `cover` | the same |
| stick / WASD | — | slides ALONG the surface; held AWAY from it for 0.30 s, leaves cover |
| **RMB** / left trigger | `aim` | leans out — around the nearer corner, or up over a low top |
| **Space** | `jump` | vaults OVER a low cover, onto the ground on its far side |

One key for both directions, GTA's own binding: a player pressing the cover key while in cover
means *the other one*. The name lives in `inf_ecs::movement::actions::COVER` and is rebindable from
the in-game settings dialog like every other control (`Take Cover`).

### The thresholds

Every number is on `inf_ecs::cover`, so a project that wants different ones retunes one module.

| number | value | what it decides |
|---|---|---|
| `MIN_COVER_HEIGHT_M` | **0.55 m** | below this a surface is not cover at all — a 0.15 m kerb never is |
| `inf_anim::MANTLE_HIGH_SPLIT_M` | **1.25 m** | at or below it the character **crouches**; above it (or with no top within 2.5 m) it **stands** |
| `SNAP_S` | 0.25 s | how long the blend into cover takes; `SLIDE_IN_S` (0.55 s) when the character arrives sprinting |
| `MAX_SNAP_M` | 1.50 m | further than this the press refuses — a snap that crossed a room would be a teleport with a ramp on it |
| `STANDOFF_M` | 0.06 m | how far the capsule's surface sits off the cover |
| `AWAY_LEAVE_S` | 0.30 s | how long the stick must be held away from the surface to leave |
| `PEEK_LATERAL_M` | 0.45 m | how far a corner peek steps the character out past the edge |

The classification is the whole of "crouching or standing depending on the object", and the split is
**ALS's own 125 cm mantle line** rather than a second number: the height at which a body stops being
able to get over a thing is the height at which it stops being able to crouch behind it.

### What the character does

Sliding along cover uses the same gaits, the same stride warping and the same foot IK a walk does —
the stick is projected onto the surface's tangent and everything downstream is unchanged. The slide
**stops at the corner** the probe measured, with the capsule's own edge on it. Where a surface
changes height along its length (a car's bonnet into a wall behind it) the class changes with it and
the character stands up.

A peek **moves the capsule**, not only the pose: around a corner it steps 0.45 m out, and over a low
top it stands up. That is deliberate and it is what makes cover mean anything — a shot fired at a
tucked-in character hits the cover, and the same shot at a peeking one reaches the head.

### For NPCs

Police and SWAT under fire use the same door. A responder inside a gunshot's radius that did not fire
it searches the compass for cover between it and the shooter, walks there along an `inf_nav::NavPath`,
and **presses the same `cover` edge a player's key raises**. At the town's top response rung
(`Response::Swat`) the units prefer HIGH cover, which is the only class you can shoot around an edge
from. An agent that is not under fire runs **zero** cover probes.

### The camera

The cover camera is a claim on the director's `Override` layer, not a second camera: it slides the
pivot toward the open side and shortens the boom, and it stops by not pushing. When the pivot itself
is inside geometry it **refuses** — it re-pushes the last legal pose rather than putting the optical
centre in a wall.

### What is not there yet

**Blind fire** — firing from behind cover without leaning out, the arm over the top or
around the edge with the head down — is not built. The peek, the aim and the lean are; the
shot is not. It is deliberately WPN2e's: that wave owns the NPC firing cadence and the
trigger, and a second override of a shot's direction written here is the thing it would
have to reconcile. The price when it comes is one authored upper-body one-shot, one branch
in the weapon step that points the shot along the cover's surface normal, and one gate arm
over the ray.

### The clips

ALS ships no cover sequence of any kind, so `inf_anim::authored` derives four from the rig that will
play them: `INF_Cover_Low_Idle`, `INF_Cover_Low_Move`, `INF_Cover_High_Idle` and
`INF_Cover_High_Move`. They are re-derived by `inf-import` whenever a character is imported, and the
machine's `cover` / `cover_side` / `cover_peek` parameters are what select and lean them.
