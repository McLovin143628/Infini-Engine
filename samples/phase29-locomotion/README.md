# Phase 29 Locomotion (the P29.6 gate scene)

An **obstacle course**, and it is a course rather than a level because of what
the movement-catalogue amendment asks of it: *P29.6's course must force every
catalogue mode in its one deterministic replay, so the (pose, mode) trace
certifies the catalogue and not a subset.* Every block in
`Phase29Locomotion.inf_lvl` exists because one mode cannot be reached without
it -- a 1.4 m roof to crouch under, four 20 cm risers to autostep up, ledges at
1 m and 3 m for the two mantle height classes, a 5 m one for the drop a landing
classifies as a ragdoll, and a 3 m pool to swim in and under.

## The character is the wizard's own output

`Hero.inf_skel`, `Hero Body.inf_mesh`, the three cycles and
`Hero Locomotion.inf_sm` are what `inf_editor_core::character::build_character`
produces from the default biped -- generated here with fixed GUIDs so the
committed bytes are reproducible, but through the same doors: the template rig,
the rig-derived locomotion set, `inf_anim::derive_clip` at the import door and
`inf_anim::propose` over what the derivation measured.

**The three cycles are the repository's first committed DERIVED content.** They
carry a root-motion track, a distance track, foot-plant sync markers, footstep
notifies and six curve channels -- none of which any committed clip had before,
which is the remainder P29.4 and P29.5 both wrote down.

## The text beside them is the point

- `Hero Locomotion.inf_sm.txt` -- the machine, as text (pillar S1). One value
per line, conditions as expressions, and `phase29_gate`'s one-line-diff arm
edits exactly one of those lines and measures what changes.
- `camera.toml` -- the locomotion camera's table. A camera is not sim state, so
it has no home in the scene schema and lives here instead.
- `input.toml` -- the bindings, in the format the shipped player already reads
beside a level.

## What the gate does with it

`runtime/inf-player/tests/phase29_gate.rs`: PIE == shipping byte-for-byte on the
(pose, mode) trace with every mode named, bit-exact replay across two
independent cooks, Blueprint-versus-transpiled parity over a course segment
driven through the `anim.*` kit, the one-line-diff demonstration, and a camera
trace that is deterministic and is NOT part of the sim trace.

Regenerate with `INF_BLESS_SAMPLES=1 cargo test -p inf-editor-core samples`.
