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
