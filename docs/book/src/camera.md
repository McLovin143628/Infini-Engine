# The Camera

Every character Infini Engine builds comes with a camera and a boom already on
it. You do not add one; you tune the one that is there.

The rig is a **`CameraRig`** component, and the numbers in it are the ported
[ALS](https://github.com/dyanikoglu/ALS-Community) table: a third-person boom
that lags on three axes at once, a per-gait length, an over-the-shoulder offset,
an aim block that pulls in and narrows the field of view, and a first-person
seat. Wave CHAR1c added the collision policy — a whisker fan, an asymmetric
smoother, a character-ignore rule and a near fade — and the director that
decides who is holding the camera when more than one thing wants it.

## Where the numbers live, and what does NOT persist

**The table an author owns is `camera.toml` beside the LEVEL.** That is the one
file the engine reads (`inf_player::input::load_camera_beside`), and every field
in it defaults, so a file naming one number is a legal file and every number it
does not name is the shipped ALS value:

```toml
# camera.toml -- only what this level's camera wants differently
[velocity_direction.run]
arm_length_m = 4.2

[collision]
near_fade_start_m = 1.2
```

A character may also carry a **`CameraRig`** component, which **overrides** that
table for as long as the camera is following it -- what makes possessing an NPC a
real change of camera rather than a change of subject. A character built through
the New Character wizard is given one, with the same ALS defaults on it.

**A rig does not persist.** It is a runtime default, not an authored value, and
the audit arm `an_authored_rig_does_not_survive_a_save_and_a_reload` is each half
of that as a number:

* the component is not in the level's bytes, so a level saved and reloaded has no
  rig on any character -- including one the wizard made a moment earlier;
* the `camera.toml` the wizard writes beside a *character asset* is the plain
  default and is never read by anything;
* a tune made with the live-tuning slider or with `camera.set_rig` is session
  state, exactly as a vehicle's or a weapon's is, and is gone at the next launch.

So: **to keep a camera change, edit the `camera.toml` beside the level.** The rig
is for a running game -- a possessed NPC, a Blueprint that wants a different boom
for a scene -- and the two doors that would let an authored rig survive (a write
half for the level table, or a character-asset table read at level load) are
filed and not built.

The rig is deliberately not in the level's own bytes: every character in every
level this engine ships wants the same table, and putting it in the scene record
would cost a schema version, a frozen entity record, both hosts' apply-record
mirrors, the play-in-editor payload version and a re-cook of every committed
level -- for 776 bytes per character carrying identical numbers.

## The boom, and what gets in its way

The camera sits at the end of a boom that starts at a **pivot** on the character
(80 % of its standing height by default, so a 1.2 m character gets a
proportionate camera without a second table) and reaches back by the current
gait's `arm_length_m`. Three things can shorten or move it.

**The main sweep.** A sphere is cast from the pivot to where the camera wants to
be, and on a blocking hit the camera stops at the contact. This is the only thing
that shortens the boom.

**The whisker fan.** A short fan of casts either side of the boom — two per side
at 22.5° by default — so the camera answers a wall it is *about* to swing into
rather than one it is already in. A blocked whisker **steers** the boom away
from itself; it does not shorten it. (A fan that also shortened took a metre off
a 3.04 m boom for a wall 45 cm to the side of a corridor, which is a camera that
bobs once per alley for geometry it was never going to touch.)

**Other characters.** The camera passes through people — its own subject, every
other character, and their ragdolls. It does **not** pass through their
vehicles: a pedestrian is something a camera may see through, two tonnes of
bodywork is not.

## Fast in, slow out

The boom comes in **on the step the world says so** and eases back out over
`return_speed`. The asymmetry is not a taste: a boom that eased *in* would be a
camera inside the wall for as long as the ease took, and no finite speed makes
that zero. Going back out is the opposite problem — a boom that sprang out the
instant a lamp-post cleared would pump once per lamp-post — so that half is
slow, and `return_speed` is the number that says how slow.

## The near fade

When the boom is shorter than `near_fade_start_m` the character's own body
starts to fade out, and below `near_fade_end_m` it is gone. It is a screen-door
dither on the skinned path rather than a visibility switch, so a body backed
into a corner thins out instead of popping. The body still writes depth and
still casts its shadow, which is what a player wants: your shadow does not
vanish because you leaned on a wall.

## The director

More than one thing can want the camera. The director resolves it with three
layers, and the order between them is fixed:

| layer | who asks | beats |
|---|---|---|
| `Gameplay` | the character's own rig, the drive camera, aiming | — |
| `Override` | a death cam, a ragdoll follow, a photo mode | gameplay |
| `Scripted` | a sequencer shot, a cutscene | everything |

A source pushes a **claim** for one step at a time; the step it stops pushing,
the camera blends back. There is no release verb, deliberately: a cutscene that
ended in an early return would otherwise own the camera for the rest of the
session. A claim with a blend time arrives over a `smoothstep`; a claim with
zero is a **cut**, and the renderer is told it was one.

## From a Blueprint

Three nodes, under `camera`:

| node | what it does |
|---|---|
| **Set Camera Rig Value** | one value by name — `run.arm_length_m`, `aim.fov_deg`, `collision.pull_in_speed`, `shoulder`, `first_person`. A character with no rig gets one and then takes the write. |
| **Get Camera Rig Value** | the same vocabulary, read back. |
| **Camera Shot** | put the camera at an entity's transform for **this step** — a scripted shot. Hold it by calling it every Tick. |

The names are the same ones `camera.toml` uses and the same ones the live tuning
slider drives, so an author who has learned one has learned all three.

## The controls

| key | what it does |
|---|---|
| right mouse | aim — pulls the camera in, narrows the field of view, and turns the body to face where you look |
| **Q** | cycle the rotation mode: face where you are *going* ↔ face where you are *looking* |
| **G** | first person ↔ third person (going first person also asks the character to face where it looks, because there is no other way to aim in first person) |

The shoulder is a rig value (`shoulder`) rather than a key: it is a decision an
author makes once. It is **not** a `camera.toml` key and **not** a live-tuning
row -- both of those speak `CameraTuning`'s vocabulary, and the shoulder is one
of the two names (`shoulder`, `first_person`) that live on the rig itself. Today
its only door is a Blueprint's `camera.set_rig`, and like every rig value it is
gone at the next launch.
